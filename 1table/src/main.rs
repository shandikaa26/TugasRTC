use std::f64::consts::PI;
use eframe::egui;

const TABLE_SIZE: usize = 360;
const DEG_TO_RAD: f64 = PI / 180.0;

// Lookup table untuk sin dan cos
fn generate_lookup_tables() -> (Vec<f64>, Vec<f64>) {
    let mut sin_table = Vec::with_capacity(TABLE_SIZE);
    let mut cos_table = Vec::with_capacity(TABLE_SIZE);

    for deg in 0..TABLE_SIZE {
        let rad = deg as f64 * DEG_TO_RAD;
        sin_table.push(rad.sin());
        cos_table.push(rad.cos());
    }

    (sin_table, cos_table)
}

// Fungsi untuk mendapatkan sin dan cos dari lookup table
fn lookup_sin_cos(deg: f64, sin_table: &Vec<f64>, cos_table: &Vec<f64>) -> (f64, f64) {
    let index = (deg.round() as usize) % TABLE_SIZE;
    (sin_table[index], cos_table[index])
}

struct TrigCalculatorApp {
    sin_table: Vec<f64>,
    cos_table: Vec<f64>,
    angle_deg: String,
    result_sin: f64,
    result_cos: f64,
    result_tan: f64,
    calculated: bool,
    error_message: String,
    show_table: bool,
    table_start: usize,
    table_end: usize,
}

impl TrigCalculatorApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (sin_table, cos_table) = generate_lookup_tables();
        Self {
            sin_table,
            cos_table,
            angle_deg: "60".to_string(),
            result_sin: 0.0,
            result_cos: 0.0,
            result_tan: 0.0,
            calculated: false,
            error_message: String::new(),
            show_table: false,
            table_start: 0,
            table_end: 10,
        }
    }
    
    fn calculate(&mut self) {
        match self.angle_deg.parse::<f64>() {
            Ok(angle) => {
                let (sin_val, cos_val) = lookup_sin_cos(angle, &self.sin_table, &self.cos_table);
                self.result_sin = sin_val;
                self.result_cos = cos_val;
                // Handle division by zero for tangent
                self.result_tan = if cos_val.abs() < 1e-10 {
                    if sin_val >= 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }
                } else {
                    sin_val / cos_val
                };
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

impl eframe::App for TrigCalculatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Kalkulator Trigonometri dengan Lookup Table");
            ui.add_space(20.0);
            
            ui.horizontal(|ui| {
                ui.label("Masukkan sudut (derajat):");
                let response = ui.text_edit_singleline(&mut self.angle_deg);
                
                if response.changed() {
                    self.calculated = false;
                }
                
                if ui.button("Hitung").clicked() || (response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    self.calculate();
                }
            });
            
            if !self.error_message.is_empty() {
                ui.colored_label(egui::Color32::RED, &self.error_message);
            }
            
            if self.calculated {
                ui.add_space(10.0);
                egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                    ui.add_space(10.0);
                    ui.heading(format!("Hasil untuk sudut {}°:", self.angle_deg));
                    ui.add_space(5.0);
                    ui.label(format!("sin({}°) = {:.6}", self.angle_deg, self.result_sin));
                    ui.label(format!("cos({}°) = {:.6}", self.angle_deg, self.result_cos));
                    ui.label(format!("tan({}°) = {:.6}", self.angle_deg, self.result_tan));
                    ui.add_space(10.0);
                });
            }
            
            ui.add_space(20.0);
            ui.separator();
            ui.add_space(10.0);
            
            ui.collapsing("Tentang Lookup Table", |ui| {
                ui.label("Aplikasi ini menggunakan lookup table untuk mendapatkan nilai trigonometri dengan cepat.");
                ui.label("Lookup table berisi nilai-nilai yang sudah dihitung sebelumnya untuk semua sudut 0-359 derajat.");
                ui.label("Metode ini lebih cepat dari perhitungan langsung karena menghindari operasi floating-point yang kompleks.");
            });
            
            ui.add_space(10.0);
            
            // Menampilkan isi lookup table
            ui.checkbox(&mut self.show_table, "Tampilkan isi lookup table");
            
            if self.show_table {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Rentang tabel:");
                    ui.add(egui::Slider::new(&mut self.table_start, 0..=359).text("Dari"));
                    ui.add(egui::Slider::new(&mut self.table_end, 0..=359).text("Sampai"));
                });
                
                // Pastikan rentang valid
                if self.table_start > self.table_end {
                    let temp = self.table_start;
                    self.table_start = self.table_end;
                    self.table_end = temp;
                }
                
                // Batasi jumlah entri yang ditampilkan
                if self.table_end - self.table_start > 50 {
                    self.table_end = self.table_start + 50;
                }
                
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("lookup_table_grid")
                        .num_columns(4)
                        .spacing([10.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Sudut (°)");
                            ui.strong("Sin");
                            ui.strong("Cos");
                            ui.strong("Tan");
                            ui.end_row();
                            
                            for deg in self.table_start..=self.table_end {
                                let sin_val = self.sin_table[deg];
                                let cos_val = self.cos_table[deg];
                                let tan_val = if cos_val.abs() < 1e-10 {
                                    if sin_val >= 0.0 { f64::INFINITY } else { f64::NEG_INFINITY }
                                } else {
                                    sin_val / cos_val
                                };
                                
                                ui.label(format!("{}", deg));
                                ui.label(format!("{:.6}", sin_val));
                                ui.label(format!("{:.6}", cos_val));
                                
                                if tan_val.is_infinite() {
                                    if tan_val.is_sign_positive() {
                                        ui.label("∞");
                                    } else {
                                        ui.label("-∞");
                                    }
                                } else {
                                    ui.label(format!("{:.6}", tan_val));
                                }
                                
                                ui.end_row();
                            }
                        });
                });
            }
            
            // Visualization of sine wave
            if self.calculated {
                ui.add_space(20.0);
                ui.heading("Visualisasi Gelombang Sinus");
                
                let angle_rad = self.angle_deg.parse::<f64>().unwrap_or(0.0) * DEG_TO_RAD;
                plot_sine_wave(ui, angle_rad);
            }
        });
    }
}

fn plot_sine_wave(ui: &mut egui::Ui, highlight_angle: f64) {
    let width = ui.available_width().min(600.0);
    let height = 200.0;
    
    let (response, painter) = ui.allocate_painter(egui::Vec2::new(width, height), egui::Sense::hover());
    let rect = response.rect;
    
    // Background
    painter.rect_filled(rect, 5.0, egui::Color32::from_rgb(30, 30, 50));
    
    // Draw axes
    let x_axis_y = rect.center().y;
    painter.line_segment(
        [rect.left_top() + egui::vec2(0.0, x_axis_y - rect.top()), 
         rect.right_top() + egui::vec2(0.0, x_axis_y - rect.top())],
        (1.0, egui::Color32::GRAY)
    );
    
    // Draw grid lines
    for i in 0..=4 {
        let x = rect.left() + (rect.width() * i as f32 / 4.0);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            (0.5, egui::Color32::from_rgba_premultiplied(100, 100, 100, 100))
        );
        
        let angle_text = format!("{}°", i * 90);
        painter.text(
            egui::pos2(x, rect.bottom() - 2.0),
            egui::Align2::CENTER_BOTTOM,
            angle_text,
            egui::FontId::default(),
            egui::Color32::LIGHT_GRAY
        );
    }
    
    // Draw sine wave
    let points_per_unit = 5.0;
    let total_points = (width * points_per_unit) as usize;
    
    let mut points = Vec::with_capacity(total_points);
    for i in 0..total_points {
        // Convert everything to f32 for egui compatibility
        let t = 2.0_f32 * PI as f32 * (i as f32 / total_points as f32);
        let x = rect.left() + rect.width() * (t / (2.0_f32 * PI as f32));
        let y = x_axis_y - (t.sin() * height * 0.4);
        points.push(egui::pos2(x, y));
    }
    
    if points.len() >= 2 {
        // Draw sine curve
        painter.add(egui::Shape::line(points, (2.0, egui::Color32::from_rgb(100, 200, 255))));
        
        // Highlight the input angle on the curve
        let normalized_angle = highlight_angle % (2.0 * PI);
        let highlight_x = rect.left() + rect.width() * (normalized_angle as f32 / (2.0 * PI) as f32);
        let highlight_y = x_axis_y - ((normalized_angle as f32).sin() * height * 0.4);
        
        // Draw vertical line to the curve
        painter.line_segment(
            [egui::pos2(highlight_x, x_axis_y), egui::pos2(highlight_x, highlight_y)],
            (1.0, egui::Color32::YELLOW)
        );
        
        // Draw highlight point
        painter.circle_filled(
            egui::pos2(highlight_x, highlight_y),
            5.0,
            egui::Color32::RED
        );
        
        // Draw angle text
        let angle_deg = (highlight_angle * 180.0 / PI) as i32;
        painter.text(
            egui::pos2(highlight_x, highlight_y - 15.0),
            egui::Align2::CENTER_BOTTOM,
            format!("{}°", angle_deg),
            egui::FontId::default(),
            egui::Color32::WHITE
        );
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 700.0])
            .with_min_inner_size([400.0, 300.0])
            .with_title("Kalkulator Trigonometri"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Kalkulator Trigonometri",
        options,
        Box::new(|cc| Box::new(TrigCalculatorApp::new(cc)))
    )
}