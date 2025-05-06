use linfa::prelude::*;
use linfa_svm::Svm;
use linfa_logistic::MultiLogisticRegression;
use ndarray::{s, Array1, Array2, Axis};
use plotters::prelude::*;
use csv::ReaderBuilder;
use std::collections::HashMap;
use std::fs;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;

// Normalisasi manual
fn normalize(data: &Array2<f64>) -> Array2<f64> {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let std = data.std_axis(Axis(0), 0.0);
    let mut normalized = data.clone();

    for mut row in normalized.outer_iter_mut() {
        for ((x, m), s) in row.iter_mut().zip(mean.iter()).zip(std.iter()) {
            *x = if *s != 0.0 { (*x - *m) / *s } else { 0.0 };
        }
    }

    normalized
}

fn load_data(path: &str) -> Result<(Array2<f64>, Array1<usize>, Vec<String>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let headers = rdr.headers()?.clone();
    let feature_names: Vec<String> = headers
        .iter()
        .filter(|&h| h.to_lowercase() != "potability")
        .take(2)
        .map(|s| s.to_string())
        .collect();

    let mut records: Vec<csv::StringRecord> = rdr.records().collect::<Result<_, csv::Error>>()?;
    records.shuffle(&mut thread_rng());

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in &records {
        let feat: Vec<f64> = record.iter()
            .take(record.len() - 1)
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
    
        if feat.len() == record.len() - 1 {
            features.push(feat);
            let label = record.get(record.len() - 1)
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(0);  
            labels.push(label);
        }
    }

    let num_features = features[0].len();
    Ok((
        Array2::from_shape_vec((features.len(), num_features), features.concat())?,
        Array1::from_vec(labels),
        feature_names,
    ))
}

fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

fn predict_knn(train: &Array2<f64>, train_labels: &Array1<usize>, test: &Array2<f64>, k: usize) -> Array1<usize> {
    let mut predictions = Array1::zeros(test.nrows());

    for (i, test_sample) in test.axis_iter(Axis(0)).enumerate() {
        let mut distances: Vec<_> = train
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(j, train_sample)| {
                let dist = euclidean_distance(&train_sample.to_owned(), &test_sample.to_owned());
                (j, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut counts = HashMap::new();
        for &(idx, _) in distances.iter().take(k) {
            let label = train_labels[idx];
            *counts.entry(label).or_insert(0) += 1;
        }

        let predicted_label = counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
        predictions[i] = predicted_label;
    }

    predictions
}

fn train_svm(train_x: Array2<f64>, train_y: Array1<f64>, test_x: Array2<f64>) -> Array1<usize> {
    let dataset = Dataset::new(train_x, train_y);
    let model = Svm::params()
        .gaussian_kernel(100.0)
        .fit(&dataset)
        .expect("Gagal melatih SVM");

    // Konversi hasil prediksi f64 menjadi usize
    model
        .predict(&test_x)
        .mapv(|x| if x > 0.5 { 1 } else { 0 })
}



fn accuracy(pred: &Array1<usize>, target: &Array1<usize>) -> f64 {
    pred.iter()
        .zip(target.iter())
        .filter(|(p, t)| *p == *t)
        .count() as f64 / pred.len() as f64
}

fn plot_svm_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    test_label: usize,
    file_name: &str,
    feature_names: (&str, &str),
) -> Result<(), Box<dyn Error>> {
    let caption = format!(
        "SVM Decision Boundary (Predicted: {})",
        match test_label {
            0 => "Unsafe",
            1 => "Safe",
            _ => "Unknown",
        }
    );

    let (feature_x, feature_y) = feature_names;

    let x_min = train.column(0).iter().chain(test.column(0).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = train.column(0).iter().chain(test.column(0).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = train.column(1).iter().chain(test.column(1).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = train.column(1).iter().chain(test.column(1).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let root = BitMapBackend::new(file_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().x_desc(feature_x).y_desc(feature_y).draw()?;

    let dataset = Dataset::new(train.clone(), train_labels.clone())
        .with_feature_names(vec![feature_x.to_string(), feature_y.to_string()]);

    let model = MultiLogisticRegression::default().max_iterations(100).fit(&dataset)?;

    let grid_size = 100;
    let x_step = (x_max - x_min) / grid_size as f64;
    let y_step = (y_max - y_min) / grid_size as f64;

    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = x_min + i as f64 * x_step;
            let y = y_min + j as f64 * y_step;
            let point = Array2::from_shape_vec((1, 2), vec![x, y])?;
            let prediction = model.predict(&point)[0];

            let color = match prediction {
                0 => RGBColor(255, 200, 200),
                1 => RGBColor(200, 255, 200),
                _ => WHITE,
            };

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + x_step, y + y_step)],
                ShapeStyle::from(&color).filled(),
            )))?;
        }
    }

    for (i, label) in train_labels.iter().enumerate() {
        let color = match label {
            0 => RED.mix(0.7),
            1 => GREEN.mix(0.7),
            _ => BLACK.into(),
        };

        chart.draw_series(PointSeries::of_element(
            vec![(train[[i, 0]], train[[i, 1]])],
            8,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
                    + Text::new(
                        match label {
                            0 => "U",
                            1 => "S",
                            _ => "?",
                        },
                        (0, 10),
                        ("sans-serif", 15).into_font(),
                    )
            },
        ))?;
    }

    let test_color = match test_label {
        0 => RED,
        1 => GREEN,
        _ => BLACK,
    };

    chart.draw_series(PointSeries::of_element(
        vec![(test[[0, 0]], test[[0, 1]])],
        15,
        ShapeStyle::from(&test_color).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new("X", (0, 15), ("sans-serif", 20).into_font())
        },
    ))?;

    root.present()?;
    println!(" Plot disimpan di '{}'", file_name);
    Ok(())
}

fn plot_knn_neighbors(
    train: &Array2<f64>,
    train_labels: &Array1<usize>,
    test: &Array2<f64>,
    test_label: usize,
    file_name: &str,
    feature_names: (&str, &str),
) -> Result<(), Box<dyn Error>> {
    let caption = format!(
        "KNN Prediction (Predicted: {})",
        match test_label {
            0 => "Unsafe",
            1 => "Safe",
            _ => "Unknown",
        }
    );

    let (feature_x, feature_y) = feature_names;

    let x_min = train.column(0).iter().chain(test.column(0).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = train.column(0).iter().chain(test.column(0).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = train.column(1).iter().chain(test.column(1).iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = train.column(1).iter().chain(test.column(1).iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let root = BitMapBackend::new(file_name, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().x_desc(feature_x).y_desc(feature_y).draw()?;

    for (i, label) in train_labels.iter().enumerate() {
        let color = match label {
            0 => RED.mix(0.7),
            1 => GREEN.mix(0.7),
            _ => BLACK.into(),
        };

        chart.draw_series(PointSeries::of_element(
            vec![(train[[i, 0]], train[[i, 1]])],
            8,
            ShapeStyle::from(&color).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
                    + Text::new(
                        match label {
                            0 => "U",
                            1 => "S",
                            _ => "?",
                        },
                        (0, 10),
                        ("sans-serif", 15).into_font(),
                    )
            },
        ))?;
    }

    let test_color = match test_label {
        0 => RED,
        1 => GREEN,
        _ => BLACK,
    };

    chart.draw_series(PointSeries::of_element(
        vec![(test[[0, 0]], test[[0, 1]])],
        15,
        ShapeStyle::from(&test_color).filled(),
        &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new("X", (0, 15), ("sans-serif", 20).into_font())
        },
    ))?;

    root.present()?;
    println!(" Plot disimpan di '{}'", file_name);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (features, labels_usize, feature_names_vec) = load_data("csv/water_potability.csv")?;
    let feature_name_pair = (feature_names_vec[0].as_str(), feature_names_vec[1].as_str());

    let label_count = labels_usize.iter().fold([0; 2], |mut acc, &x| {
        acc[x] += 1;
        acc
    });
    println!(" Distribusi label: Unsafe = {}, Safe = {}", label_count[0], label_count[1]);

    let split_idx = (features.nrows() as f64 * 0.8) as usize;
    let (train_x, test_x) = features.view().split_at(Axis(0), split_idx);
    let (train_y_usize, test_y_usize) = labels_usize.view().split_at(Axis(0), split_idx);
    let train_y_f64 = train_y_usize.to_owned().mapv(|x| x as f64);


    // Normalisasi sebelum training
    let train_x = normalize(&train_x.to_owned());
    let test_x = normalize(&test_x.to_owned());

    let svm_preds = train_svm(train_x.clone(), train_y_f64, test_x.clone());
    let svm_acc = accuracy(&svm_preds, &test_y_usize.to_owned());
    println!(" Akurasi SVM: {:.2}%", svm_acc * 100.0);

    let knn_preds = predict_knn(&train_x, &train_y_usize.to_owned(), &test_x, 3);
    let knn_acc = accuracy(&knn_preds, &test_y_usize.to_owned());
    println!(" Akurasi KNN: {:.2}%", knn_acc * 100.0);

    println!("\n 20 Prediksi SVM vs Aktual:");
    for (i, (pred, actual)) in svm_preds.iter().zip(test_y_usize.iter()).take(20).enumerate() {
        println!("Data {:2}: Prediksi = {}, Aktual = {}", i + 1, pred, actual);
    }

    println!("\n 20 Prediksi KNN vs Aktual:");
    for (i, (pred, actual)) in knn_preds.iter().zip(test_y_usize.iter()).take(20).enumerate() {
        println!("Data {:2}: Prediksi = {}, Aktual = {}", i + 1, pred, actual);
    }

    fs::create_dir_all("output")?;
    plot_svm_neighbors(
        &train_x.slice(s![.., 0..2]).to_owned(),
        &train_y_usize.to_owned(),
        &test_x.slice(s![0..1, 0..2]).to_owned(),
        svm_preds[0] as usize,
        "output/svm_plot.png",
        feature_name_pair
    )?;
        plot_knn_neighbors(
        &train_x.slice(s![.., 0..2]).to_owned(),
        &train_y_usize.to_owned(),
        &test_x.slice(s![0..1, 0..2]).to_owned(),
        knn_preds[0] as usize,
        "output/knn_plot.png",
        feature_name_pair
    )?;


    Ok(())
}
