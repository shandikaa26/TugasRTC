use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::fs;
use rfd::FileDialog;

// Training parameters struct to share between threads
#[derive(Clone, Debug)]
pub struct TrainingParams {
    pub epochs: usize,
    pub hidden_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f64,
    pub restart_training: bool,
}

// Dataset configuration struct
#[derive(Clone, Debug)]
pub struct DatasetConfig {
    pub path: String,
    pub has_headers: bool,
    pub delimiter: char,
    pub label_column: usize,
    pub skip_rows_with_errors: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            has_headers: true,
            delimiter: ',',
            label_column: 0, // Will be updated after dataset inspection
            skip_rows_with_errors: true,
        }
    }
}

pub struct TrainingWindow {
    accuracies: Vec<f64>,
    losses: Vec<f64>,
    receiver: Receiver<(f64, f64)>,
    training_params: Arc<Mutex<TrainingParams>>,
    params_sender: Sender<Arc<Mutex<TrainingParams>>>,
    dataset_config: Arc<Mutex<DatasetConfig>>,
    dataset_sender: Sender<Arc<Mutex<DatasetConfig>>>,
    epochs_input: String,
    hidden_layers_input: String,
    neurons_input: String,
    learning_rate_input: String,
    is_training: bool,
    last_received_time: std::time::Instant,
    training_completed: bool,
    first_run: bool,
    dataset_preview: String,
    column_count: usize,
    label_column_input: String,
    available_columns: Vec<String>,
    error_message: String,
    success_message: String,
    has_dataset: bool,
}

impl TrainingWindow {
    pub fn new() -> (Self, Sender<(f64, f64)>, Receiver<Arc<Mutex<TrainingParams>>>, Receiver<Arc<Mutex<DatasetConfig>>>) {
        let (accuracy_sender, accuracy_receiver) = channel();
        let (params_sender, params_receiver) = channel();
        let (dataset_sender, dataset_receiver) = channel();
        
        let training_params = Arc::new(Mutex::new(TrainingParams {
            epochs: 2000,
            hidden_layers: 2,
            neurons_per_layer: 32,
            learning_rate: 0.5,
            restart_training: false,
        }));
        
        let dataset_config = Arc::new(Mutex::new(DatasetConfig::default()));
        
        (Self {
            accuracies: Vec::new(),
            losses: Vec::new(),
            receiver: accuracy_receiver,
            training_params: training_params.clone(),
            params_sender,
            dataset_config: dataset_config.clone(),
            dataset_sender,
            epochs_input: "2000".to_string(),
            hidden_layers_input: "2".to_string(),
            neurons_input: "32".to_string(),
            learning_rate_input: "0.5".to_string(),
            is_training: false,
            last_received_time: std::time::Instant::now(),
            training_completed: false,
            first_run: true,
            dataset_preview: String::new(),
            column_count: 0,
            label_column_input: "0".to_string(),
            available_columns: Vec::new(),
            error_message: String::new(),
            success_message: String::new(),
            has_dataset: false,
        }, accuracy_sender, params_receiver, dataset_receiver)
    }
    
    fn browse_for_dataset(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("CSV Files", &["csv"])
            .add_filter("All Files", &["*"])
            .set_title("Select Dataset CSV File")
            .pick_file() 
        {
            self.load_dataset_preview(&path);
        }
    }
    
    fn load_dataset_preview(&mut self, path: &PathBuf) {
        // Reset messages
        self.error_message.clear();
        self.success_message.clear();
        
        // Read the first few lines of the CSV file for preview
        match fs::read_to_string(path) {
            Ok(content) => {
                // Extract just the first 5 lines for preview
                let preview_lines: Vec<&str> = content.lines().take(5).collect();
                self.dataset_preview = preview_lines.join("\n");
                
                // Try to determine delimiter and count columns
                let first_line = preview_lines.first().unwrap_or(&"");
                
                // Try to guess the delimiter
                let delimiter = self.guess_delimiter(first_line);
                
                // Count columns based on the guessed delimiter
                let cols = first_line.split(delimiter).count();
                self.column_count = cols;
                
                // Update available columns if we have a header
                let has_headers = {
                    let config = self.dataset_config.lock().unwrap();
                    config.has_headers
                };
                
                if has_headers && !preview_lines.is_empty() {
                    self.available_columns = first_line.split(delimiter)
                        .map(|s| s.trim().to_string())
                        .collect();
                } else {
                    // No headers, just use column indices
                    self.available_columns = (0..cols)
                        .map(|i| format!("Column {}", i))
                        .collect();
                }
                
                // Set the label column to the last column by default
                if !self.available_columns.is_empty() {
                    let last_col_idx = self.available_columns.len() - 1;
                    self.label_column_input = last_col_idx.to_string();
                    
                    // Update dataset config
                    {
                        let mut config = self.dataset_config.lock().unwrap();
                        config.path = path.to_string_lossy().to_string();
                        config.delimiter = delimiter;
                        config.label_column = last_col_idx;
                    }
                    
                    // Auto-apply configuration - send to training thread automatically
                    self.dataset_sender.send(self.dataset_config.clone()).unwrap_or_else(|e| {
                        self.error_message = format!("Failed to send dataset configuration: {}", e);
                    });
                    
                    self.has_dataset = true; // This is critical for enabling the training button
                    self.success_message = format!("Dataset loaded: {} with {} columns\nLabel column automatically set to '{}'", 
                                                  path.file_name().unwrap_or_default().to_string_lossy(), 
                                                  cols,
                                                  self.available_columns[last_col_idx]);
                }
            },
            Err(e) => {
                self.error_message = format!("Failed to load dataset: {}", e);
                self.dataset_preview = String::new();
                self.has_dataset = false;
            }
        }
    }
    
    fn guess_delimiter(&self, line: &str) -> char {
        // Try to guess the delimiter by counting occurrences
        let comma_count = line.matches(',').count();
        let tab_count = line.matches('\t').count();
        let semicolon_count = line.matches(';').count();
        
        // Return the most frequent delimiter
        if comma_count >= tab_count && comma_count >= semicolon_count {
            ','
        } else if tab_count >= comma_count && tab_count >= semicolon_count {
            '\t'
        } else {
            ';'
        }
    }
    
    fn update_dataset_config(&mut self) {
        // Make sure we have a valid label column index
        if let Ok(label_col) = self.label_column_input.parse::<usize>() {
            if label_col < self.column_count {
                let mut config = self.dataset_config.lock().unwrap();
                config.label_column = label_col;
                
                // Send the updated config to the training thread
                self.dataset_sender.send(self.dataset_config.clone()).unwrap_or_else(|e| {
                    self.error_message = format!("Failed to send dataset configuration: {}", e);
                });
                
                self.success_message = "Dataset configuration updated successfully".to_string();
            } else {
                self.error_message = format!("Invalid label column index. Must be between 0 and {}", self.column_count - 1);
            }
        } else {
            self.error_message = "Label column must be a valid number".to_string();
        }
    }
}

impl eframe::App for TrainingWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Simple non-blocking check for new updates - only tries once per frame
        match self.receiver.try_recv() {
            Ok((accuracy, loss)) => {
                // Got an update, add to charts
                self.accuracies.push(accuracy);
                self.losses.push(loss);
                self.is_training = true;
                self.first_run = false;
                self.training_completed = false;
                self.last_received_time = std::time::Instant::now();
                
                // Keep charts from growing too large
                if self.accuracies.len() > 500 {
                    self.accuracies.remove(0);
                    self.losses.remove(0);
                }
            },
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // No updates this frame - check if training might be complete
                if self.is_training && 
                   self.last_received_time.elapsed() > std::time::Duration::from_secs(2) &&
                   !self.training_completed {
                    self.is_training = false;
                    self.training_completed = true;
                }
            },
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Channel closed, training thread must have exited
                if self.is_training {
                    self.is_training = false;
                    self.training_completed = true;
                }
            }
        }

        // Dataset Selection Panel
        egui::TopBottomPanel::top("dataset_panel").show(ctx, |ui| {
            ui.heading("Dataset Selection");
            ui.add_space(10.0);
            
            ui.horizontal(|ui| {
                if ui.button("Browse for Dataset").clicked() {
                    self.browse_for_dataset();
                }
                
                let config_path = {
                    let config = self.dataset_config.lock().unwrap();
                    config.path.clone()
                };
                
                if !config_path.is_empty() {
                    ui.label(&config_path);
                } else {
                    ui.label("No dataset selected");
                }
            });
            
            // Show dataset configuration options
            if !self.dataset_preview.is_empty() {
                ui.collapsing("Dataset Preview", |ui| {
                    ui.monospace(&self.dataset_preview);
                });
                
                ui.add_space(5.0);
                
                // Headers checkbox
                {
                    let mut has_headers = {
                        let config = self.dataset_config.lock().unwrap();
                        config.has_headers
                    };
                    
                    ui.horizontal(|ui| {
                        if ui.checkbox(&mut has_headers, "First row contains headers").changed() {
                            {
                                let mut config = self.dataset_config.lock().unwrap();
                                config.has_headers = has_headers;
                            }
                            
                            // Need to capture the current path
                            let current_path = {
                                let config = self.dataset_config.lock().unwrap();
                                PathBuf::from(&config.path)
                            };
                            
                            // Update column list if headers changed
                            if !current_path.as_os_str().is_empty() {
                                self.load_dataset_preview(&current_path);
                            }
                        }
                        
                        // Delimiter selection
                        ui.label("Delimiter:");
                        
                        let (current_delimiter, delimiters, delimiter_labels) = {
                            let config = self.dataset_config.lock().unwrap();
                            (config.delimiter, [',', ';', '\t', '|'], ["Comma (,)", "Semicolon (;)", "Tab (\\t)", "Pipe (|)"])
                        };
                        
                        let mut selected_idx = delimiters.iter()
                            .position(|&d| d == current_delimiter)
                            .unwrap_or(0);
                        
                        egui::ComboBox::from_id_source("delimiter_combo")
                            .selected_text(delimiter_labels[selected_idx])
                            .show_ui(ui, |ui| {
                                for (idx, &label) in delimiter_labels.iter().enumerate() {
                                    if ui.selectable_label(selected_idx == idx, label).clicked() {
                                        selected_idx = idx;
                                        {
                                            let mut config = self.dataset_config.lock().unwrap();
                                            config.delimiter = delimiters[idx];
                                        }
                                        
                                        // Need to capture the current path
                                        let current_path = {
                                            let config = self.dataset_config.lock().unwrap();
                                            PathBuf::from(&config.path)
                                        };
                                        
                                        // Update preview with new delimiter
                                        if !current_path.as_os_str().is_empty() {
                                            self.load_dataset_preview(&current_path);
                                        }
                                    }
                                }
                            });
                    });
                }
                
                ui.add_space(5.0);
                
                // Label column selection
                ui.horizontal(|ui| {
                    ui.label("Select Label Column:");
                    
                    egui::ComboBox::from_id_source("label_column_combo")
                        .selected_text(if !self.available_columns.is_empty() {
                            if let Ok(idx) = self.label_column_input.parse::<usize>() {
                                if idx < self.available_columns.len() {
                                    &self.available_columns[idx]
                                } else {
                                    "Invalid column"
                                }
                            } else {
                                "Invalid column"
                            }
                        } else {
                            "No columns available"
                        })
                        .show_ui(ui, |ui| {
                            for (idx, col_name) in self.available_columns.iter().enumerate() {
                                if ui.selectable_label(self.label_column_input == idx.to_string(), col_name).clicked() {
                                    self.label_column_input = idx.to_string();
                                }
                            }
                        });
                    
                    if ui.button("Apply Configuration").clicked() {
                        self.update_dataset_config();
                    }
                });
                
                // Error handling option
                ui.horizontal(|ui| {
                    let mut config = self.dataset_config.lock().unwrap();
                    ui.checkbox(&mut config.skip_rows_with_errors, "Skip rows with errors");
                    ui.label("(Uncheck to replace errors in feature columns with 0.0)");
                });
            }
            
            // Display error or success messages
            if !self.error_message.is_empty() {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::RED, &self.error_message);
                    if ui.button("Clear").clicked() {
                        self.error_message.clear();
                    }
                });
            }
            
            if !self.success_message.is_empty() {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::GREEN, &self.success_message);
                    if ui.button("Clear").clicked() {
                        self.success_message.clear();
                    }
                });
            }
        });

        // Neural Network Parameters Panel
        egui::TopBottomPanel::top("parameters_panel").show(ctx, |ui| {
            ui.heading("Neural Network Training Parameters");
            ui.add_space(10.0);
            
            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.text_edit_singleline(&mut self.epochs_input);
                
                ui.label("Hidden Layers:");
                ui.text_edit_singleline(&mut self.hidden_layers_input);
                
                ui.label("Neurons per Layer:");
                ui.text_edit_singleline(&mut self.neurons_input);
                
                ui.label("Learning Rate:");
                ui.text_edit_singleline(&mut self.learning_rate_input);
            });
            
            ui.horizontal(|ui| {
                let button_text = if self.first_run {
                    "Start Training"
                } else if self.training_completed {
                    "Restart Training with New Parameters"
                } else if self.is_training {
                    "Update Parameters After Training"
                } else {
                    "Start Training"
                };
                
                // Add button, but add more safety checks
                let training_enabled = !self.is_training && (!self.dataset_preview.is_empty());
                let button = ui.add_enabled(training_enabled, egui::Button::new(button_text));
                
                if button.clicked() {
                    // Use catch_unwind to prevent potential crash
                    let result = std::panic::catch_unwind(|| {
                        // Safety check - ensure we actually have a dataset before proceeding
                        if self.dataset_preview.is_empty() {
                            return Err("No dataset loaded. Please select a dataset first.".to_string());
                        }
                        
                        // Make sure we have some data in available_columns
                        if self.available_columns.is_empty() {
                            return Err("Dataset appears to be empty or incorrectly formatted.".to_string());
                        }
                        
                        // Try to parse all parameters
                        let epochs = self.epochs_input.parse::<usize>()
                            .map_err(|_| "Invalid epoch value".to_string())?;
                        if epochs == 0 {
                            return Err("Epochs must be greater than 0".to_string());
                        }
                        
                        let hidden_layers = self.hidden_layers_input.parse::<usize>()
                            .map_err(|_| "Invalid hidden layers value".to_string())?;
                        if hidden_layers == 0 {
                            return Err("Hidden layers must be at least 1".to_string());
                        }
                        
                        let neurons = self.neurons_input.parse::<usize>()
                            .map_err(|_| "Invalid neurons value".to_string())?;
                        if neurons == 0 {
                            return Err("Neurons per layer must be at least 1".to_string());
                        }
                        
                        let lr = self.learning_rate_input.parse::<f64>()
                            .map_err(|_| "Invalid learning rate".to_string())?;
                        if lr <= 0.0 {
                            return Err("Learning rate must be greater than 0".to_string());
                        }
                        
                        // All parameters are valid
                        Ok((epochs, hidden_layers, neurons, lr))
                    });
                    
                    // Handle potential panic
                    match result {
                        Ok(params_result) => {
                            match params_result {
                                Ok((epochs, hidden_layers, neurons, lr)) => {
                                    // Clear any previous error messages
                                    self.error_message.clear();
                                    
                                    // Safely lock parameters
                                    match self.training_params.lock() {
                                        Ok(mut params) => {
                                            // Set the parameters
                                            params.epochs = epochs;
                                            params.hidden_layers = hidden_layers;
                                            params.neurons_per_layer = neurons;
                                            params.learning_rate = lr;
                                            params.restart_training = true;
                                            
                                            // Clone the parameters for sending
                                            let params_clone = self.training_params.clone();
                                            
                                            // Clear charts
                                            self.accuracies.clear();
                                            self.losses.clear();
                                            
                                            // Update state before sending to avoid race condition
                                            self.is_training = true;
                                            self.training_completed = false;
                                            
                                            // Send parameters to training thread
                                            match self.params_sender.send(params_clone) {
                                                Ok(_) => {
                                                    self.success_message = "Training started successfully.".to_string();
                                                },
                                                Err(e) => {
                                                    self.error_message = format!("Failed to send parameters: {}", e);
                                                    self.is_training = false; // Reset state
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            self.error_message = format!("Failed to lock training parameters: {}", e);
                                        }
                                    }
                                },
                                Err(err_msg) => {
                                    self.error_message = err_msg.clone();
                                }
                            }
                        },
                        Err(_) => {
                            self.error_message = "Internal error occurred.".to_string();
                        }
                    }
                }
                
                // Display disabled state reason
                if self.is_training {
                    ui.colored_label(egui::Color32::YELLOW, "Training in progress...");
                } else if self.dataset_preview.is_empty() {
                    ui.colored_label(egui::Color32::RED, "Please select a dataset first");
                }
            });
            
            ui.add_space(5.0);
            
            // Display status
            if self.is_training {
                ui.horizontal(|ui| {
                    ui.label("🔄 Training in progress...");
                    if let Some(&last_accuracy) = self.accuracies.last() {
                        if let Some(&last_loss) = self.losses.last() {
                            ui.label(format!("Current Accuracy: {:.2}%, Loss: {:.4}", last_accuracy, last_loss));
                        }
                    }
                });
            } else if self.training_completed {
                ui.horizontal(|ui| {
                    ui.label("✅ Training completed.");
                    if let Some(&last_accuracy) = self.accuracies.last() {
                        if let Some(&last_loss) = self.losses.last() {
                            ui.label(format!("Final Accuracy: {:.2}%, Loss: {:.4}", last_accuracy, last_loss));
                        }
                    }
                });
                ui.label("You can change parameters and restart training.");
            } else if self.first_run {
                ui.label("👆 Set parameters and click 'Start Training' to begin");
            } else {
                ui.label("⏸️ Training not active. Click the button to start.");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Simplify the layout to ensure plots are visible
            ui.vertical(|ui| {
                let available_height = ui.available_height();
                
                // Accuracy plot with green line
                ui.heading("Accuracy (%)");
                Plot::new("accuracy_plot")
                    .height(available_height * 0.4)
                    .show_axes(true)
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show(ui, |plot_ui| {
                        if !self.accuracies.is_empty() {
                            // Convert accuracies to points
                            let points: PlotPoints = self.accuracies
                                .iter()
                                .enumerate()
                                .map(|(i, &acc)| [i as f64, acc])
                                .collect();
                            
                            // Create a line from the points with green color
                            let line = Line::new(points)
                                .name("Accuracy (%)")
                                .width(2.0)
                                .color(egui::Color32::from_rgb(50, 205, 50)); // Green
                            
                            // Add the line to the plot
                            plot_ui.line(line);
                            
                            // Set the plot bounds
                            let max_y = self.accuracies.iter().fold(0.0f64, |a, &b| a.max(b)).max(1.0);
                            plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                [0.0, 0.0],
                                [self.accuracies.len() as f64, max_y * 1.1],
                            ));
                        } else {
                            // If no data yet, show a message in the plot area
                            plot_ui.text(
                                egui_plot::Text::new(
                                    egui_plot::PlotPoint::new(0.5, 0.5),
                                    "Accuracy data will appear here"
                                )
                            );
                        }
                    });
                
                ui.add_space(10.0); // Add some space between plots
                
                // Loss plot with red line
                ui.heading("Loss");
                Plot::new("loss_plot")
                    .height(available_height * 0.4)
                    .show_axes(true)
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show(ui, |plot_ui| {
                        if !self.losses.is_empty() {
                            // Convert losses to points
                            let points: PlotPoints = self.losses
                                .iter()
                                .enumerate()
                                .map(|(i, &loss)| [i as f64, loss])
                                .collect();
                            
                            // Create a line from the points with red color
                            let line = Line::new(points)
                                .name("Loss")
                                .width(2.0)
                                .color(egui::Color32::from_rgb(220, 50, 50)); // Red
                            
                            // Add the line to the plot
                            plot_ui.line(line);
                            
                            // Set the plot bounds
                            let max_y = self.losses.iter().fold(0.0f64, |a, &b| a.max(b)).max(1.0);
                            plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                [0.0, 0.0],
                                [self.losses.len() as f64, max_y * 1.1],
                            ));
                        } else {
                            // If no data yet, show a message in the plot area
                            plot_ui.text(
                                egui_plot::Text::new(
                                    egui_plot::PlotPoint::new(0.5, 0.5),
                                    "Loss data will appear here"
                                )
                            );
                        }
                    });
            });
        });
    }
} 