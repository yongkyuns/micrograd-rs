use micrograd_rs::prelude::*;
use plotters::prelude::*;

const OUT_FILE_NAME: &str = "binary_classifier.png";

fn main() {
    let model = MLP::new(2, vec![16, 16, 1], Activation::Relu);
    let data = moon::moon_data();

    for k in 0..100 {
        let (total_loss, acc) = loss(&model, &data, None);
        model.zero_grad();
        total_loss.backward();

        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.0;

        for p in model.parameters().iter() {
            p.step(learning_rate);
        }

        if k % 1 == 0 {
            println!(
                "Step {:2.2} | loss {:3.3} | accuracy {:5.1}%\n",
                k,
                total_loss.data(),
                acc * 100.0
            );
        }
    }
    plot(&model).unwrap();
}

fn plot(model: &MLP) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 100;
    const RANGE: f64 = 5.0;
    const PIXEL_WIDTH: f64 = RANGE / N as f64;

    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 1024)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-RANGE / 2.0..RANGE / 2.0, -RANGE / 2.0..RANGE / 2.0)?;

    chart
        .configure_mesh()
        .x_label_style(TextStyle::from(("sans-serif", 25)))
        .y_label_style(TextStyle::from(("sans-serif", 25)))
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    let mut matrix = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let x = i as f64 * PIXEL_WIDTH - RANGE / 2.0;
            let y = j as f64 * PIXEL_WIDTH - RANGE / 2.0;
            let score = model.run(vec![Value::new(x), Value::new(y)])[0].data();
            matrix[j][i] = score;
        }
    }

    chart.draw_series(
        matrix
            .iter()
            .zip(0..)
            .flat_map(|(l, y)| l.iter().zip(0..).map(move |(v, x)| (x, y, v)))
            .map(|(i, j, v)| {
                let x = i as f64 * PIXEL_WIDTH - RANGE / 2.0;
                let y = j as f64 * PIXEL_WIDTH - RANGE / 2.0;
                Rectangle::new(
                    [(x, y), (x + PIXEL_WIDTH, y + PIXEL_WIDTH)],
                    RGBAColor(
                        if *v > 0.0 { 255 } else { 0 },
                        0,
                        if *v > 0.0 { 0 } else { 255 },
                        0.4,
                    )
                    .filled(),
                )
            }),
    )?;

    let points = moon::moon_data();

    chart.draw_series(points.iter().map(|(x, y, v)| {
        if *v > 0 {
            Circle::new((*x, *y), 7, RED.filled())
        } else {
            Circle::new((*x, *y), 7, BLUE.filled())
        }
    }))?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())
}

fn loss(model: &MLP, data: &Vec<(f64, f64, i32)>, batch_size: Option<usize>) -> (Value, f64) {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    let batch_size = batch_size.unwrap_or(data.len());
    let batch: Vec<_> = data
        .choose_multiple(&mut thread_rng(), batch_size)
        .collect();

    let scores = batch
        .iter()
        .map(|&(a, b, _)| model.run(vec![Value::new(*a), Value::new(*b)]))
        .collect::<Vec<_>>();

    let losses = batch
        .iter()
        .map(|&(_, _, yb)| yb)
        .zip(scores.iter())
        .map(|(yi, scorei)| {
            let scorei = scorei[0].clone();
            (scorei * (-yi as f64) + 1.0).relu()
        })
        .collect::<Vec<_>>();

    let data_loss = losses
        .iter()
        .fold(Value::new(0.0), |sum, loss| sum + loss.clone())
        * (1.0 / losses.len() as f64);

    let alpha = 1.0e-4;
    let reg_loss = model
        .parameters()
        .iter()
        .map(|p| p.clone().pow(Value::new(2.0)))
        .fold(Value::new(0.0), |sum, p| sum + p)
        * alpha;
    let total_loss = data_loss + reg_loss;

    let result = batch
        .into_iter()
        .map(|&(_, _, yb)| yb)
        .zip(scores.iter())
        .map(|(yi, scorei)| (yi > 0) == (scorei[0].data() > 0.0))
        .collect::<Vec<_>>();

    let accuracy =
        result.iter().filter(|&&r| r).fold(0.0, |acc, _| acc + 1.0) / result.len() as f64;

    (total_loss, accuracy)
}
