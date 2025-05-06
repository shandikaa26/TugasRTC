use std::f64::consts::PI;
use eframe::egui;

// Faktorial
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

// Fungsi sine menggunakan deret Taylor
fn taylor_sin(x: f64, terms: u32) -> f64 {
    let mut result = 0.0;
    let mut sign = 1.0;

    for n in 0..terms {
        let power = 2 * n + 1;
        result += sign * x.powi(power as i32) / factorial(power as u64) as f64;
        sign *= -1.0;
    }

    result
}

// Fungsi cosine menggunakan deret Taylor
fn taylor_cos(x: f64, terms: u32) -> f64 {
    let mut result = 0.0;
    let mut sign = 1.0;

    for n in 0..terms {
        let power = 2 * n;
        result += sign * x.powi(power as i32) / factorial(power as u64) as f64;
        sign *= -1.0;
    }

    result
}

// Fungsi tangent menggunakan sin/cos
fn taylor_tan(x: f64, terms: u32) -> f64 {
    let sin_val = taylor_sin(x, terms);
    let cos_val = taylor_cos(x, terms);
    
    if cos_val.abs() < 1e-10 {
        if sin_val >= 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        sin_val / cos_val
    }
}

struct TaylorSeriesApp {
    angle_deg: String,
    terms: u32,
    rad_mode: bool,
    result_sin: f64,
    result_cos: f64,
    result_tan: f64,
    builtin_sin: f64,
    builtin_cos: f64,
    builtin_tan: f64,
    calculated: bool,
    error_message: String,
    show_terms: bool,
    show_comparison: bool,
}

impl TaylorSeriesApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            angle_deg: "45".to_string(),
            terms: 10,
            rad_mode: false,
            result_sin: 0.0,
            result_cos: 0.0,
            result_tan: 0.0,
            builtin_sin: 0.0,
            builtin_cos: 0.0,
            builtin_tan: 0.0,
            calculated: false,
            error_message: String::new(),
            show_terms: false,
            show_comparison: true,
        }
    }
    
    fn calculate(&mut self) {
        match self.angle_deg.parse::<f64>() {
            Ok(angle) => {
                // Convert to radians if in degree mode
                let angle_rad = if self.rad_mode {
                    angle
                } else {
                    angle * PI / 180.0
                };
                
                // Calculate using Taylor series
                self.result_sin = taylor_sin(angle_rad, self.terms);
                self.result_cos = taylor_cos(angle_rad, self.terms);
                self.result_tan = taylor_tan(angle_rad, self.terms);
                
                // Calculate using built-in functions for comparison
                self.builtin_sin = angle_rad.sin();
                self.builtin_cos = angle_rad.cos();
                self.builtin_tan = angle_rad.tan();
                
                self.calculated = true;
                self.error_message = String::new();
            },
            Err(_) => {
                self.error_message = "Mohon masukkan nilai sudut yang valid".to_string();
                self.calculated = false;
            }
        }
    }
}

impl eframe::App for TaylorSeriesApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Kalkulator Deret Taylor untuk Fungsi Trigonometri");
            ui.add_space(20.0);
            
            ui.horizontal(|ui| {
                ui.label("Masukkan sudut:");
                let response = ui.text_edit_singleline(&mut self.angle_deg);
                
                ui.label(if self.rad_mode { "radian" } else { "derajat" });
                
                ui.checkbox(&mut self.rad_mode, "Mode Radian");
                
                if response.changed() {
                    self.calculated = false;
                }
            });
            
            ui.horizontal(|ui| {
                ui.label("Jumlah suku deret:");
                if ui.add(egui::Slider::new(&mut self.terms, 1..=50)).changed() {
                    self.calculated = false;
                }
            });
            
            if ui.button("Hitung").clicked() {
                self.calculate();
            }
            
            if !self.error_message.is_empty() {
                ui.colored_label(egui::Color32::RED, &self.error_message);
            }
            
            if self.calculated {
                ui.add_space(10.0);
                egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                    ui.add_space(10.0);
                    ui.heading(format!("Hasil untuk sudut {} {}:", 
                                       self.angle_deg, 
                                       if self.rad_mode { "rad" } else { "°" }));
                    ui.add_space(5.0);
                    
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.strong("Dengan Deret Taylor:");
                            ui.label(format!("sin = {:.10}", self.result_sin));
                            ui.label(format!("cos = {:.10}", self.result_cos));
                            ui.label(format!("tan = {:.10}", self.result_tan));
                        });
                        
                        if self.show_comparison {
                            ui.separator();
                            ui.vertical(|ui| {
                                ui.strong("Fungsi Bawaan Rust:");
                                ui.label(format!("sin = {:.10}", self.builtin_sin));
                                ui.label(format!("cos = {:.10}", self.builtin_cos));
                                ui.label(format!("tan = {:.10}", self.builtin_tan));
                            });
                            
                            ui.separator();
                            ui.vertical(|ui| {
                                ui.strong("Selisih Absolut:");
                                ui.label(format!("sin: {:.10e}", (self.result_sin - self.builtin_sin).abs()));
                                ui.label(format!("cos: {:.10e}", (self.result_cos - self.builtin_cos).abs()));
                                ui.label(format!("tan: {:.10e}", (self.result_tan - self.builtin_tan).abs()));
                            });
                        }
                    });
                    
                    ui.add_space(5.0);
                    ui.checkbox(&mut self.show_comparison, "Tampilkan perbandingan dengan fungsi bawaan");
                    ui.add_space(5.0);
                });
            }
            
            ui.add_space(20.0);
            ui.separator();
            
            ui.collapsing("Tentang Deret Taylor", |ui| {
                ui.label("Deret Taylor adalah metode untuk menghitung nilai fungsi sebagai deret tak hingga.");
                ui.label("Untuk sin(x):");
                ui.monospace("sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...");
                ui.label("Untuk cos(x):");
                ui.monospace("cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...");
                ui.label("Semakin banyak suku yang digunakan, semakin akurat hasilnya.");
            });
            
            // Suku-suku deret yang digunakan dalam perhitungan
            ui.add_space(10.0);
            ui.checkbox(&mut self.show_terms, "Tampilkan suku-suku deret");
            
            if self.show_terms && self.calculated {
                ui.add_space(10.0);
                
                match self.angle_deg.parse::<f64>() {
                    Ok(angle) => {
                        let angle_rad = if self.rad_mode {
                            angle
                        } else {
                            angle * PI / 180.0
                        };
                        
                        egui::Grid::new("taylor_terms_grid")
                            .num_columns(4)
                            .spacing([10.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("n");
                                ui.strong("Suku sin(x)");
                                ui.strong("Suku cos(x)");
                                ui.strong("Nilai parsial");
                                ui.end_row();
                                
                                let mut partial_sin = 0.0;
                                let mut partial_cos = 0.0;
                                let mut sign_sin = 1.0;
                                let mut sign_cos = 1.0;
                                
                                for n in 0..self.terms {
                                    // Sin term
                                    let power_sin = 2 * n + 1;
                                    let term_sin = sign_sin * angle_rad.powi(power_sin as i32) / 
                                                 factorial(power_sin as u64) as f64;
                                    partial_sin += term_sin;
                                    
                                    // Cos term
                                    let power_cos = 2 * n;
                                    let term_cos = sign_cos * angle_rad.powi(power_cos as i32) / 
                                                 factorial(power_cos as u64) as f64;
                                    partial_cos += term_cos;
                                    
                                    // Display
                                    ui.label(format!("{}", n));
                                    
                                    let sin_term_text = format!("{}{:.10}",
                                        if sign_sin > 0.0 { "+" } else { "-" },
                                        term_sin.abs());
                                    ui.label(sin_term_text);
                                    
                                    let cos_term_text = format!("{}{:.10}",
                                        if sign_cos > 0.0 { "+" } else { "-" },
                                        term_cos.abs());
                                    ui.label(cos_term_text);
                                    
                                    ui.label(format!("sin={:.10}, cos={:.10}", partial_sin, partial_cos));
                                    ui.end_row();
                                    
                                    // Flip signs for next terms
                                    sign_sin *= -1.0;
                                    sign_cos *= -1.0;
                                }
                            });
                    },
                    Err(_) => {
                        ui.colored_label(egui::Color32::RED, "Tidak dapat menampilkan suku-suku dengan nilai sudut yang tidak valid");
                    }
                }
            }
            
            // Visualization
            if self.calculated {
                ui.add_space(20.0);
                ui.heading("Visualisasi Fungsi Trigonometri");
                
                match self.angle_deg.parse::<f64>() {
                    Ok(angle) => {
                        let angle_rad = if self.rad_mode {
                            angle
                        } else {
                            angle * PI / 180.0
                        };
                        plot_trig_functions(ui, angle_rad);
                    },
                    Err(_) => {}
                }
            }
        });
    }
}

fn plot_trig_functions(ui: &mut egui::Ui, highlight_angle: f64) {
    let width = ui.available_width().min(600.0);
    let height = 200.0;
    
    let (response, painter) = ui.allocate_painter(egui::Vec2::new(width, height), egui::Sense::hover());
    let rect = response.rect;
    
    // Background
    painter.rect_filled(rect, 5.0, egui::Color32::from_rgb(30, 30, 50));
    
    // Horizontal axis (y = 0)
    let y_axis = rect.center().y;
    painter.line_segment(
        [egui::pos2(rect.left(), y_axis), egui::pos2(rect.right(), y_axis)],
        (1.0, egui::Color32::GRAY)
    );
    
    // Vertical gridlines at multiples of π
    for i in -2..=2 {
        let x_pos = rect.center().x + (i as f32 * rect.width() / 4.0);
        if x_pos >= rect.left() && x_pos <= rect.right() {
            painter.line_segment(
                [egui::pos2(x_pos, rect.top()), egui::pos2(x_pos, rect.bottom())],
                (0.5, egui::Color32::from_rgba_premultiplied(100, 100, 100, 100))
            );
            
            // Label
            let label = match i {
                -2 => "-π",
                -1 => "-π/2",
                0 => "0",
                1 => "π/2",
                2 => "π",
                _ => "",
            };
            
            painter.text(
                egui::pos2(x_pos, rect.bottom() - 5.0),
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::default(),
                egui::Color32::LIGHT_GRAY
            );
        }
    }
    
    // Draw sine curve (blue)
    let points_per_unit = 5.0;
    let total_points = (width * points_per_unit) as usize;
    
    let mut sin_points = Vec::with_capacity(total_points);
    let mut cos_points = Vec::with_capacity(total_points);
    
    for i in 0..total_points {
        let t = (i as f32 / total_points as f32) * (4.0 * std::f32::consts::PI) - (2.0 * std::f32::consts::PI);
        let x = rect.center().x + (t * rect.width() / (4.0 * std::f32::consts::PI));
        
        // Sin
        let sin_y = y_axis - (t.sin() * height * 0.4);
        sin_points.push(egui::pos2(x, sin_y));
        
        // Cos
        let cos_y = y_axis - (t.cos() * height * 0.4);
        cos_points.push(egui::pos2(x, cos_y));
    }
    
    // Draw curves
    if sin_points.len() >= 2 {
        painter.add(egui::Shape::line(sin_points, (2.0, egui::Color32::from_rgb(100, 150, 255))));
    }
    
    if cos_points.len() >= 2 {
        painter.add(egui::Shape::line(cos_points, (2.0, egui::Color32::from_rgb(255, 150, 100))));
    }
    
    // Highlight the input angle on both curves
    let normalized_angle = ((highlight_angle + 2.0 * PI) % (4.0 * PI)) - 2.0 * PI;
    
    let x_highlight = rect.center().x + (normalized_angle as f32 * rect.width() / (4.0 * std::f32::consts::PI));
    let sin_y_highlight = y_axis - ((normalized_angle as f32).sin() * height * 0.4);
    let cos_y_highlight = y_axis - ((normalized_angle as f32).cos() * height * 0.4);
    
    // Draw vertical line from axis to the point
    painter.line_segment(
        [egui::pos2(x_highlight, y_axis), egui::pos2(x_highlight, sin_y_highlight)],
        (1.0, egui::Color32::YELLOW)
    );
    
    // Draw points on the curves
    painter.circle_filled(egui::pos2(x_highlight, sin_y_highlight), 5.0, egui::Color32::BLUE);
    painter.circle_filled(egui::pos2(x_highlight, cos_y_highlight), 5.0, egui::Color32::from_rgb(255, 150, 100));
    
    // Draw angle text and legend
    let angle_deg = (highlight_angle * 180.0 / PI) as i32;
    painter.text(
        egui::pos2(x_highlight, rect.top() + 15.0),
        egui::Align2::CENTER_TOP,
        format!("{}°", angle_deg),
        egui::FontId::default(),
        egui::Color32::WHITE
    );
    
    // Legend
    let legend_x = rect.right() - 80.0;
    let legend_y = rect.top() + 20.0;
    painter.circle_filled(egui::pos2(legend_x, legend_y), 4.0, egui::Color32::from_rgb(100, 150, 255));
    painter.text(
        egui::pos2(legend_x + 10.0, legend_y),
        egui::Align2::LEFT_CENTER,
        "sin(x)",
        egui::FontId::default(),
        egui::Color32::from_rgb(100, 150, 255)
    );
    
    painter.circle_filled(egui::pos2(legend_x, legend_y + 20.0), 4.0, egui::Color32::from_rgb(255, 150, 100));
    painter.text(
        egui::pos2(legend_x + 10.0, legend_y + 20.0),
        egui::Align2::LEFT_CENTER,
        "cos(x)",
        egui::FontId::default(),
        egui::Color32::from_rgb(255, 150, 100)
    );
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 800.0])
            .with_min_inner_size([400.0, 300.0])
            .with_title("Kalkulator Deret Taylor"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Kalkulator Deret Taylor",
        options,
        Box::new(|cc| Box::new(TaylorSeriesApp::new(cc)))
    )
}