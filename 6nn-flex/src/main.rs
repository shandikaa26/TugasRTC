use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use rand::thread_rng;
use rand::seq::SliceRandom;
use csv::ReaderBuilder;
use std::error::Error;
use std::thread;
use std::sync::mpsc::{Sender, TryRecvError};
use std::path::Path;
mod frontend_qt;
use frontend_qt::{TrainingWindow, TrainingParams, DatasetConfig};

// Logging interval for training progress
const LOG_INTERVAL: usize = 100;

fn shuffle_data(x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut indices: Vec<usize> = (0..x.nrows()).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let x_shuffled = Array2::from_shape_fn(x.raw_dim(), |(i, j)| x[(indices[i], j)]);
    let y_shuffled = Array2::from_shape_fn(y.raw_dim(), |(i, j)| y[(indices[i], j)]);
    (x_shuffled, y_shuffled)
}

fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

fn relu_deriv(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn normalize(mut data: Array2<f64>) -> Array2<f64> {
    for mut col in data.columns_mut() {
        let mean = col.mean().unwrap();
        let std = col.mapv(|x| (x - mean).powi(2)).mean().unwrap().sqrt();
        col -= mean;
        col /= std.max(1e-8);
    }
    data
}

fn load_data(path: &str, config: &DatasetConfig) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error + Send + Sync>> {
    // Verify file exists
    if !Path::new(path).exists() {
        return Err(format!("Dataset file not found: {}", path).into());
    }

    let mut rdr = ReaderBuilder::new()
        .has_headers(config.has_headers)
        .delimiter(config.delimiter as u8)
        .from_path(path)?;
    
    // Handle files with headers: gather column names for logging
    let headers: Vec<String> = if config.has_headers {
        if let Some(headers) = rdr.headers().ok() {
            headers.iter().map(|s| s.to_string()).collect()
        } else {
            return Err("Failed to read headers from CSV file".into());
        }
    } else {
        Vec::new() // No headers
    };
    
    // Check if label index is valid
    let total_columns = if !headers.is_empty() {
        headers.len()
    } else {
        // If no headers, we need to peek at the first record
        let mut peek_rdr = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(config.delimiter as u8)
            .from_path(path)?;
        
        if let Some(Ok(record)) = peek_rdr.records().next() {
            record.len()
        } else {
            return Err("Failed to read first record to determine column count".into());
        }
    };
    
    if config.label_column >= total_columns {
        return Err(format!(
            "Invalid label column index {}. File has {} columns (0-indexed)",
            config.label_column, total_columns
        ).into());
    }
    
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    
    // Set to track which columns have parsing errors
    let mut error_columns: Vec<bool> = vec![false; total_columns];
    let mut _total_rows = 0;
    let mut _valid_rows = 0;
    
    // Process each record
    for (_row_idx, result) in rdr.records().enumerate() {
        _total_rows += 1;
        
        match result {
            Ok(record) => {
                // Skip empty rows
                if record.len() == 0 {
                    continue;
                }
                
                let mut feature_row: Vec<f64> = Vec::new();
                let mut has_errors = false;
                let mut label_value: Option<f64> = None;
                
                // Process each field
                for (col_idx, field) in record.iter().enumerate() {
                    // Try to parse the field as a float
                    match field.trim().parse::<f64>() {
                        Ok(value) => {
                            if col_idx == config.label_column {
                                // This is our label column
                                label_value = Some(value);
                            } else {
                                // This is a feature column
                                feature_row.push(value);
                            }
                        },
                        Err(_) => {
                            // Mark this column as having an error
                            error_columns[col_idx] = true;
                            has_errors = true;
                            
                            // If this is the label column, we can't use this row
                            if col_idx == config.label_column {
                                break;
                            }
                            
                            // For feature columns, we might replace with a default value
                            if config.skip_rows_with_errors {
                                has_errors = true;
                                break;
                            } else {
                                // Replace with 0.0 (or another strategy like mean imputation could be used)
                                feature_row.push(0.0);
                            }
                        }
                    }
                }
                
                // If we have a valid row with no errors (or we're allowing errors in feature columns)
                if !has_errors && label_value.is_some() {
                    labels.push(label_value.unwrap());
                    features.push(feature_row);
                    _valid_rows += 1;
                }
            },
            Err(_e) => {
                // Silently ignore row errors
            }
        }
    }
    
    // Log any parsing issues - but only if serious
    if error_columns.iter().filter(|&&x| x).count() > total_columns / 2 {
        // More than half of columns have errors - this might be a problem
        eprintln!("Warning: More than half of columns contain non-numeric values");
    }
    
    // Check if we have enough data
    if features.is_empty() || labels.is_empty() {
        return Err("No valid data rows found in the dataset".into());
    }
    
    // Check if labels contain only binary values (0 and 1)
    let has_non_binary = labels.iter().any(|&v| v != 0.0 && v != 1.0);
    if has_non_binary {
        eprintln!("Warning: Label column contains non-binary values. The model works best with binary classification (0 or 1).");
        
        // Attempt to binarize the labels by thresholding
        for label in labels.iter_mut() {
            *label = if *label > 0.5 { 1.0 } else { 0.0 };
        }
    }
    
    // Convert to ndarray format
    let n_features = features[0].len();
    let feature_array = Array2::from_shape_vec((features.len(), n_features), features.concat())?;
    let label_array = Array2::from_shape_vec((labels.len(), 1), labels)?;

    Ok((feature_array, label_array))
}

// Simplified train_network function with better UI responsiveness
fn train_network(
    x: &Array2<f64>, 
    y_true: &Array2<f64>, 
    params: &TrainingParams,
    sender: &Sender<(f64, f64)>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Less verbose startup message
    eprintln!("Training started");
    
    // Basic input validation
    let (n_samples, n_features) = x.dim();
    
    if n_samples == 0 || n_features == 0 {
        return Err("Empty training data".into());
    }
    
    if y_true.nrows() != n_samples {
        return Err("Mismatched data and label dimensions".into());
    }
    
    // Guard against bad parameters
    if params.hidden_layers == 0 {
        return Err("Number of hidden layers must be at least 1".into());
    }
    
    if params.neurons_per_layer == 0 {
        return Err("Number of neurons per layer must be at least 1".into());
    }
    
    // Initialize weights and biases for variable number of layers
    let mut rng = thread_rng();
    let mut weights = Vec::new();
    let mut biases = Vec::new();
    
    // Less verbose network initialization
    eprintln!("Network: {} inputs, {} hidden layers, {} neurons/layer", 
             n_features, params.hidden_layers, params.neurons_per_layer);
    
    // Input layer -> first hidden layer
    weights.push(Array2::random_using((n_features, params.neurons_per_layer), StandardNormal, &mut rng) * 0.1);
    biases.push(Array2::zeros((1, params.neurons_per_layer)));
    
    // Hidden layers
    for _ in 1..params.hidden_layers {
        weights.push(Array2::random_using(
            (params.neurons_per_layer, params.neurons_per_layer), 
            StandardNormal, 
            &mut rng
        ) * 0.1);
        biases.push(Array2::zeros((1, params.neurons_per_layer)));
    }
    
    // Output layer
    weights.push(Array2::random_using((params.neurons_per_layer, 1), StandardNormal, &mut rng) * 0.1);
    biases.push(Array2::zeros((1, 1)));
    
    // Storage for activations and gradients during training
    let mut activations = Vec::with_capacity(params.hidden_layers);
    let mut z_values = Vec::with_capacity(params.hidden_layers);
    let mut dw = Vec::with_capacity(params.hidden_layers + 1);
    let mut db = Vec::with_capacity(params.hidden_layers + 1);
    
    // Initialize gradient storage
    for i in 0..=params.hidden_layers {
        dw.push(Array2::zeros(weights[i].dim()));
        db.push(Array2::zeros(biases[i].dim()));
    }
    
    // Send initial update to start the UI
    let _ = sender.send((0.0, 0.0));
    
    // Increase log interval to reduce console spam
    let log_interval = params.epochs / 20; // Only show ~20 log messages
    let log_interval = log_interval.max(LOG_INTERVAL); // But not less than the minimum
    
    // Training loop
    let total_epochs = params.epochs;
    for epoch in 0..total_epochs {
        // Clear activations for this epoch
        activations.clear();
        z_values.clear();
        
        // Forward pass
        let mut a_prev = x.clone();
        
        // Hidden layers
        for i in 0..params.hidden_layers {
            let z = a_prev.dot(&weights[i]) + &biases[i];
            let a = relu(&z);
            
            z_values.push(z);
            activations.push(a.clone());
            a_prev = a;
        }
        
        // Output layer
        let z_out = a_prev.dot(&weights[params.hidden_layers]) + &biases[params.hidden_layers];
        let y_pred = sigmoid(&z_out);
        
        // Compute loss (binary cross-entropy)
        let epsilon = 1e-15;
        let y_pred_clipped = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let losses = y_true * &y_pred_clipped.mapv(|v| v.ln()) + 
                    (1.0 - y_true) * &y_pred_clipped.mapv(|v| (1.0 - v).ln());
        let loss = -losses.mean().unwrap_or(0.0);
        
        // Backpropagation
        // Output layer gradient
        let mut delta = &y_pred - y_true;
        
        // Update output layer weights and biases
        let a_last = if params.hidden_layers > 0 {
            &activations[params.hidden_layers - 1]
        } else {
            x
        };
        
        dw[params.hidden_layers] = a_last.t().dot(&delta) / n_samples as f64;
        let db_layer = delta.sum_axis(Axis(0)) / n_samples as f64;
        db[params.hidden_layers] = db_layer.into_shape(biases[params.hidden_layers].raw_dim()).unwrap();
        
        // Backpropagate through hidden layers
        for i in (0..params.hidden_layers).rev() {
            delta = delta.dot(&weights[i+1].t()) * relu_deriv(&z_values[i]);
            
            let input = if i == 0 {
                x
            } else {
                &activations[i-1]
            };
            
            dw[i] = input.t().dot(&delta) / n_samples as f64;
            let db_layer = delta.sum_axis(Axis(0)) / n_samples as f64;
            db[i] = db_layer.into_shape(biases[i].raw_dim()).unwrap();
        }
        
        // Update weights and biases with learning rate
        let lr = if epoch < params.epochs / 10 {
            params.learning_rate  // High learning rate for first 10%
        } else if epoch < params.epochs / 2 {
            params.learning_rate * 0.5  // Medium learning rate until halfway
        } else {
            params.learning_rate * 0.1  // Low learning rate for final half
        };
        
        for i in 0..=params.hidden_layers {
            weights[i] -= &(dw[i].clone() * lr);
            biases[i] -= &(db[i].clone() * lr);
        }
        
        // Calculate accuracy for monitoring
        let pred_labels = y_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
        let correct = pred_labels
            .iter()
            .zip(y_true.iter())
            .filter(|(p, y)| (*p - *y).abs() < 1e-6)
            .count();
        let accuracy = correct as f64 / n_samples as f64;
        
        // Log progress at fixed intervals - using adaptive interval
        if epoch % log_interval == 0 || epoch == params.epochs - 1 {
            // Simpler progress message
            eprintln!("Epoch {}/{}: Acc = {:.1}%, Loss = {:.4}", 
                     epoch, params.epochs, accuracy * 100.0, loss);
            
            // Send update to UI - non-blocking, don't care if it fails
            let _ = sender.send((accuracy * 100.0, loss));
            
            // Brief pause to let UI thread catch up
            if epoch % (log_interval * 2) == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    // Training completed successfully - simpler message
    eprintln!("Training complete");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    eprintln!("Starting application");
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    
    let (window, accuracy_sender, params_receiver, dataset_path_receiver) = TrainingWindow::new();
    
    eprintln!("Starting training thread");
    
    // Create a safer training thread with extensive error handling
    let training_thread = thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
        eprintln!("Training thread started");
        
        // Storage for variables outside the catch_unwind blocks to avoid UnwindSafe issues
        let mut current_dataset_path = String::new();
        let mut dataset_loaded = false;
        let mut x = Array2::<f64>::zeros((0, 0));
        let mut y_true = Array2::<f64>::zeros((0, 0));
        
        // Safe forever loop with error handling
        loop {
            // Try catch everything to prevent thread crash, but make sure to move variables inside the closure
            let result = std::panic::catch_unwind(|| {
                // Use local copies to avoid capturing external mutables in the closure
                let curr_path = current_dataset_path.clone();
                let is_dataset_loaded = dataset_loaded;
                
                // Check for new dataset selection first
                if let Ok(dataset_config) = dataset_path_receiver.try_recv() {
                    eprintln!("Received dataset configuration");
                    
                    // Safely unwrap the mutex, with error handling
                    let dataset_guard = match dataset_config.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            eprintln!("Error locking dataset config: {}", e);
                            return (curr_path, is_dataset_loaded, x.clone(), y_true.clone(), None);
                        }
                    };
                    
                    let new_path = dataset_guard.path.clone();
                    eprintln!("Dataset path: {}", new_path);
                    
                    // Only reload if the dataset path has changed
                    if new_path != curr_path || !is_dataset_loaded {
                        eprintln!("Loading new dataset");
                        
                        match load_data(&new_path, &dataset_guard) {
                            Ok((new_x, new_y)) => {
                                eprintln!("Dataset loaded successfully: {}x{} samples", 
                                          new_x.nrows(), new_x.ncols());
                                
                                // Update our data using a try block to avoid panics
                                let result = std::panic::catch_unwind(|| {
                                    let normalized_x = normalize(new_x);
                                    eprintln!("Data normalized");
                                    let (shuffled_x, shuffled_y) = shuffle_data(&normalized_x, &new_y);
                                    eprintln!("Data shuffled");
                                    (shuffled_x, shuffled_y)
                                });
                                
                                match result {
                                    Ok((processed_x, processed_y)) => {
                                        return (new_path, true, processed_x, processed_y, None);
                                    },
                                    Err(e) => {
                                        eprintln!("Error processing dataset: {:?}", e);
                                        return (curr_path, false, x.clone(), y_true.clone(), None);
                                    }
                                }
                            },
                            Err(e) => {
                                eprintln!("Error loading dataset: {}", e);
                                return (curr_path, false, x.clone(), y_true.clone(), None);
                            }
                        }
                    }
                }
                
                // Check for training parameters (using try_recv instead of blocking call)
                match params_receiver.try_recv() {
                    Ok(new_params) => {
                        eprintln!("Received training parameters");
                        
                        // Safely unwrap the mutex, with error handling
                        let new_params_guard = match new_params.lock() {
                            Ok(guard) => guard,
                            Err(e) => {
                                eprintln!("Error locking training parameters: {}", e);
                                return (curr_path, is_dataset_loaded, x.clone(), y_true.clone(), None);
                            }
                        };
                        
                        // If restart requested, start training
                        if new_params_guard.restart_training {
                            eprintln!("Restart training requested with params: epochs={}, hidden_layers={}, neurons={}, lr={}", 
                                      new_params_guard.epochs, new_params_guard.hidden_layers, 
                                      new_params_guard.neurons_per_layer, new_params_guard.learning_rate);
                            
                            // Create a copy for training
                            let params = TrainingParams {
                                epochs: new_params_guard.epochs,
                                hidden_layers: new_params_guard.hidden_layers,
                                neurons_per_layer: new_params_guard.neurons_per_layer,
                                learning_rate: new_params_guard.learning_rate,
                                restart_training: new_params_guard.restart_training,
                            };
                            
                            // Verify dataset is loaded before training
                            if !is_dataset_loaded || x.nrows() == 0 || y_true.nrows() == 0 {
                                eprintln!("Cannot start training: No valid dataset loaded");
                                return (curr_path, is_dataset_loaded, x.clone(), y_true.clone(), None);
                            }
                            
                            eprintln!("Starting neural network training with data: {}x{}", x.nrows(), x.ncols());
                            
                            // Start training with received parameters
                            // Wrap in a catch_unwind to prevent thread panic
                            let result = std::panic::catch_unwind(|| {
                                // Minimal version of train_network to test if it works
                                let (n_samples, n_features) = x.dim();
                                eprintln!("Data dimensions: samples={}, features={}", n_samples, n_features);
                                
                                // Initialize random weights and biases for testing
                                eprintln!("Initializing network with {} hidden layers, {} neurons", 
                                         params.hidden_layers, params.neurons_per_layer);
                                
                                // Just try to create a dummy network to check if it works
                                let mut rng = thread_rng();
                                let w = Array2::random_using((n_features, params.neurons_per_layer), StandardNormal, &mut rng);
                                let b: Array2<f64> = Array2::zeros((1, params.neurons_per_layer));
                                
                                // Try a single forward pass to detect issues
                                let z = x.dot(&w) + &b;
                                let a = relu(&z);
                                
                                eprintln!("Forward pass successful: activation shape {}x{}", a.nrows(), a.ncols());
                                
                                // Checking y_true dimensions
                                eprintln!("Label dimensions: {}x{}", y_true.nrows(), y_true.ncols());
                                
                                // If everything looks good, start actual training
                                train_network(&x, &y_true, &params, &accuracy_sender)
                            });
                            
                            return (curr_path, is_dataset_loaded, x.clone(), y_true.clone(), Some(result));
                        }
                    },
                    Err(TryRecvError::Empty) => {
                        // No message, continue
                    },
                    Err(TryRecvError::Disconnected) => {
                        // Channel closed, application is terminating
                        eprintln!("Training thread shutting down - UI disconnected");
                        std::process::exit(0);
                    }
                }
                
                // Return unchanged values if no processing was done
                (curr_path, is_dataset_loaded, x.clone(), y_true.clone(), None)
            });
            
            // Handle any panic in the main loop
            match result {
                Ok((new_path, new_dataset_loaded, new_x, new_y, training_result)) => {
                    // Update our variables outside the closure
                    current_dataset_path = new_path;
                    dataset_loaded = new_dataset_loaded;
                    x = new_x;
                    y_true = new_y;
                    
                    // Handle training result if it exists
                    if let Some(train_result) = training_result {
                        match train_result {
                            Ok(Ok(_)) => {
                                eprintln!("Training completed successfully");
                            },
                            Ok(Err(e)) => {
                                eprintln!("Error during training: {}", e);
                            },
                            Err(e) => {
                                eprintln!("Training thread panicked: {:?}", e);
                            }
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Main loop panicked: {:?}", e);
                    // Continue the outer loop to recover from panic
                }
            }
            
            // Brief sleep to avoid busy-waiting
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    eprintln!("Starting UI");
    
    // Let's handle potential errors from eframe more gracefully
    if let Err(e) = eframe::run_native(
        "Neural Network Flexible Dataset Analyzer",
        options,
        Box::new(|_cc| Box::new(window)),
    ) {
        eprintln!("Error in UI: {}", e);
    }
    
    // Join the training thread but don't panic if it fails
    if let Err(e) = training_thread.join() {
        eprintln!("Error in training thread: {:?}", e);
    }

    eprintln!("Application shutting down");
    Ok(())
}

