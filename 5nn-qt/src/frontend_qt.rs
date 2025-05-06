use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

// Training parameters struct to share between threads
#[derive(Clone, Debug)]
pub struct TrainingParams {
    pub epochs: usize,
    pub hidden_layers: usize,
    pub neurons_per_layer: usize,
    pub learning_rate: f64,
    pub restart_training: bool,
}

pub struct TrainingWindow {
    accuracies: Vec<f64>,
    losses: Vec<f64>,
    receiver: Receiver<(f64, f64)>,  // Changed to receive (accuracy, loss) tuple
    training_params: Arc<Mutex<TrainingParams>>,
    params_sender: Sender<Arc<Mutex<TrainingParams>>>,
    epochs_input: String,
    hidden_layers_input: String,
    neurons_input: String,
    learning_rate_input: String,
    is_training: bool,
    last_received_time: std::time::Instant,
    training_completed: bool,
    first_run: bool,  // Track if this is the first run
}

impl TrainingWindow {
    pub fn new() -> (Self, Sender<(f64, f64)>, Receiver<Arc<Mutex<TrainingParams>>>) {
        let (sender, receiver) = channel();
        let (params_sender, params_receiver) = channel();
        
        let training_params = Arc::new(Mutex::new(TrainingParams {
            epochs: 2000,
            hidden_layers: 2,
            neurons_per_layer: 32,
            learning_rate: 0.5,
            restart_training: false,
        }));
        
        (Self {
            accuracies: Vec::new(),
            losses: Vec::new(),
            receiver,
            training_params: training_params.clone(),
            params_sender,
            epochs_input: "2000".to_string(),
            hidden_layers_input: "2".to_string(),
            neurons_input: "32".to_string(),
            learning_rate_input: "0.5".to_string(),
            is_training: false,
            last_received_time: std::time::Instant::now(),
            training_completed: false,
            first_run: true,
        }, sender, params_receiver)
    }
}

impl eframe::App for TrainingWindow {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for new accuracy values
        let mut received_data = false;
        while let Ok((accuracy, loss)) = self.receiver.try_recv() {
            self.accuracies.push(accuracy);
            self.losses.push(loss);
            self.is_training = true;
            self.training_completed = false;
            self.first_run = false;  // No longer the first run
            self.last_received_time = std::time::Instant::now();
            received_data = true;
        }
        
        // Check if training has completed (no updates for 2 seconds)
        if self.is_training && !received_data && 
           self.last_received_time.elapsed() > std::time::Duration::from_secs(2) &&
           !self.training_completed {
            self.is_training = false;
            self.training_completed = true;
            println!("UI detected training completion");
        }

        egui::TopBottomPanel::top("parameters_panel").show(ctx, |ui| {
            ui.heading("Neural Network Water Potability Training");
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
                
                if ui.button(button_text).clicked() && !self.is_training {
                    if let Ok(parsed_epochs) = self.epochs_input.parse::<usize>() {
                        if let Ok(parsed_hidden_layers) = self.hidden_layers_input.parse::<usize>() {
                            if let Ok(parsed_neurons) = self.neurons_input.parse::<usize>() {
                                if let Ok(parsed_lr) = self.learning_rate_input.parse::<f64>() {
                                    // Parameter validation
                                    if parsed_hidden_layers == 0 {
                                        ui.label("Hidden layers must be at least 1");
                                        return;
                                    }
                                    if parsed_neurons == 0 {
                                        ui.label("Neurons per layer must be at least 1");
                                        return;
                                    }
                                    if parsed_lr <= 0.0 {
                                        ui.label("Learning rate must be greater than 0");
                                        return;
                                    }
                                    
                                    let mut params = self.training_params.lock().unwrap();
                                    params.epochs = parsed_epochs;
                                    params.hidden_layers = parsed_hidden_layers;
                                    params.neurons_per_layer = parsed_neurons;
                                    params.learning_rate = parsed_lr;
                                    params.restart_training = true;
                                    self.accuracies.clear();
                                    self.losses.clear();
                                    self.is_training = false;
                                    self.training_completed = false;
                                    
                                    // Send updated parameters to the training thread
                                    self.params_sender.send(self.training_params.clone()).unwrap_or_else(|e| {
                                        println!("Failed to send parameters: {}", e);
                                    });
                                }
                            }
                        }
                    }
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
                            ui.label(format!("Epoch: {}/{}", self.accuracies.len(), 
                                            self.training_params.lock().unwrap().epochs));
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
                            let max_y = self.losses.iter().fold(0.0f64, |a, &b| a.max(b)).max(0.1);
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
        
        // Request continuous repainting while training
        ctx.request_repaint();
    }
} 