use std::env;
use std::thread;

mod neural_network;
mod data_loader;
use crate::neural_network::{train_network, make_prediction, PredictionResult, TrainingParams};

// Import Qt integration 
mod qt_integration;
use qt_integration::NeuralNetworkQt;
use qt_integration::run_qt_app;

fn main() {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    // Set up channels for communication
    let (sender, receiver) = std::sync::mpsc::channel();
    let (params_sender, params_receiver) = std::sync::mpsc::channel();
    let (prediction_sender, prediction_receiver) = std::sync::mpsc::channel();
    let (water_params_sender, water_params_receiver) = std::sync::mpsc::channel();
    
    // Start neural network training in background thread
    thread::spawn(move || {
        println!("Starting neural network training thread");
        
        // Initialize neural network
        let mut network = neural_network::create_network();
        let data = data_loader::load_water_data("data/water_potability.csv");
        
        // Listen for parameter updates
        while let Ok(params) = params_receiver.recv() {
            let params = params.lock().unwrap().clone();
            println!("Received new training parameters: {:?}", params);
            
            if params.restart_training {
                // Train network with new parameters
                train_network(&mut network, &data, params.epochs, params.hidden_layers, 
                              params.neurons_per_layer, params.learning_rate, 
                              |epoch, accuracy, loss| {
                                  // Only send update every few epochs to avoid overwhelming UI
                                  if epoch % 10 == 0 || epoch == 1 {
                                      sender.send((accuracy, loss)).unwrap_or_else(|_| {
                                          println!("Failed to send training update");
                                      });
                                  }
                              });
            }
        }
    });
    
    // Start prediction handling thread
    thread::spawn(move || {
        println!("Starting prediction handling thread");
        
        // Initialize neural network for predictions
        let network = neural_network::create_network();
        
        // Listen for prediction requests
        while let Ok(water_params) = water_params_receiver.recv() {
            println!("Received prediction request: {:?}", water_params);
            
            // Make prediction
            match make_prediction(&network, &water_params) {
                Ok(result) => {
                    prediction_sender.send(Ok(result)).unwrap_or_else(|_| {
                        println!("Failed to send prediction result");
                    });
                },
                Err(e) => {
                    prediction_sender.send(Err(e.to_string())).unwrap_or_else(|_| {
                        println!("Failed to send prediction error");
                    });
                }
            }
        }
    });
    
    // Launch Qt application
    println!("Launching Qt application");
    run_qt_app(args);
} 