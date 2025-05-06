// === IMPORTS ===
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use rand::{thread_rng, seq::SliceRandom};
use csv::ReaderBuilder;
use std::error::Error;
use plotters::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

// === CONSTANTS ===
const EPOCHS: usize = 2000;
const LR: f64 = 0.5;
const HIDDEN: usize = 32;
const LOG_INTERVAL: usize = 100;

// === UTILS ===

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

fn shuffle_data(x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut indices: Vec<usize> = (0..x.nrows()).collect();
    let mut rng = thread_rng();
    indices.shuffle(&mut rng);

    let x_shuffled = Array2::from_shape_fn(x.raw_dim(), |(i, j)| x[(indices[i], j)]);
    let y_shuffled = Array2::from_shape_fn(y.raw_dim(), |(i, j)| y[(indices[i], j)]);
    (x_shuffled, y_shuffled)
}

// === DATA LOADER ===

fn load_data(path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let vals: Result<Vec<f64>, _> = record.iter().map(|s| s.trim().parse()).collect();
        if let Ok(vals) = vals {
            if vals.len() == 10 {
                let (x, y) = vals.split_at(9);
                features.push(x.to_vec());
                labels.push(y[0]);
            }
        }
    }

    let feature_array = Array2::from_shape_vec((features.len(), 9), features.concat())?;
    let label_array = Array2::from_shape_vec((labels.len(), 1), labels)?;
    Ok((feature_array, label_array))
}

// === PLOTTING ===

fn plot_accuracy(accuracies: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("accuracy.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_acc = accuracies.iter().copied().fold(0.0, f64::max).max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Accuracy", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..EPOCHS, 0.0..max_acc)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        accuracies.iter().enumerate().map(|(i, &acc)| (i, acc)),
        &BLUE,
    ))?;

    Ok(())
}

// === MAIN ===

fn main() -> Result<(), Box<dyn Error>> {
    let (x_raw, y_true) = load_data("src/water_potability.csv")?;
    let x = normalize(x_raw);
    let (x, y_true) = shuffle_data(&x, &y_true);
    let (n_samples, n_features) = x.dim();

    // Init model weights
    let mut rng = thread_rng();
    let mut w1 = Array2::random_using((n_features, HIDDEN), StandardNormal, &mut rng);
    let mut b1 = Array2::zeros((1, HIDDEN));
    let mut w2 = Array2::random_using((HIDDEN, HIDDEN), StandardNormal, &mut rng);
    let mut b2 = Array2::zeros((1, HIDDEN));
    let mut w3 = Array2::random_using((HIDDEN, 1), StandardNormal, &mut rng);
    let mut b3 = Array2::zeros((1, 1));

    let mut accuracies = Vec::new();
    let mut final_pred = Array2::zeros((n_samples, 1));

    // Progress bar setup
    let pb = ProgressBar::new(EPOCHS as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Epochs | Accuracy: {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    // === TRAINING LOOP ===
    for epoch in 0..EPOCHS {
        // Forward
        let z1 = x.dot(&w1) + &b1;
        let a1 = relu(&z1);
        let z2 = a1.dot(&w2) + &b2;
        let a2 = relu(&z2);
        let z3 = a2.dot(&w3) + &b3;
        let y_pred = sigmoid(&z3);

        // Backward
        let dz3 = &y_pred - &y_true;
        let dw3 = a2.t().dot(&dz3) / n_samples as f64;
        let db3 = dz3.sum_axis(Axis(0)) / n_samples as f64;

        let da2 = dz3.dot(&w3.t());
        let dz2 = da2 * relu_deriv(&z2);
        let dw2 = a1.t().dot(&dz2) / n_samples as f64;
        let db2 = dz2.sum_axis(Axis(0)) / n_samples as f64;

        let da1 = dz2.dot(&w2.t());
        let dz1 = da1 * relu_deriv(&z1);
        let dw1 = x.t().dot(&dz1) / n_samples as f64;
        let db1 = dz1.sum_axis(Axis(0)) / n_samples as f64;

        // Update
        w3 -= &(dw3 * LR);
        b3 -= &(db3 * LR);
        w2 -= &(dw2 * LR);
        b2 -= &(db2 * LR);
        w1 -= &(dw1 * LR);
        b1 -= &(db1 * LR);

        final_pred = y_pred.clone();

        // Accuracy
        let pred_labels = y_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
        let correct = pred_labels
            .iter()
            .zip(y_true.iter())
            .filter(|(p, y)| (*p - *y).abs() < 1e-6)
            .count();
        let accuracy = correct as f64 / n_samples as f64;
        accuracies.push(accuracy);

        if epoch % LOG_INTERVAL == 0 {
            println!("Epoch {}: Accuracy = {:.4}", epoch, accuracy);
        }

        pb.set_message(format!("{:.2}%", accuracy * 100.0));
        pb.inc(1);
    }

    pb.finish_with_message("Training complete");

    // Plotting & Final Accuracy
    plot_accuracy(&accuracies)?;

    let pred_labels = final_pred.mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });
    let correct = pred_labels
        .iter()
        .zip(y_true.iter())
        .filter(|(p, y)| (*p - *y).abs() < 1e-6)
        .count();

    println!("\nTotal data: {}", n_samples);
    println!("Benar terprediksi: {}", correct);
    println!("Akurasi akhir: {:.2}%", (correct as f64 / n_samples as f64) * 100.0);

    Ok(())
}
