use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Tanh,
    Relu,
    Exp,
    Pow,
    None,
}

impl Op {
    fn backward(&self, out: &Value_, a: &ValueRef, b: &ValueRef) {
        match self {
            Op::Add => {
                if let Ok(mut a) = a.write() {
                    a.grad += 1.0 * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
                if let Ok(mut b) = b.write() {
                    b.grad += 1.0 * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Mul => {
                if let (Ok(mut a), Ok(b)) = (a.write(), b.read()) {
                    a.grad += b.data * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
                if let (Ok(mut b), Ok(a)) = (b.write(), a.read()) {
                    b.grad += a.data * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Tanh => {
                if let Ok(mut a) = a.write() {
                    let tanh = out.data;
                    a.grad += (1.0 - tanh * tanh) * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Relu => {
                if let Ok(mut a) = a.write() {
                    if out.data > 0.0 {
                        a.grad += out.grad;
                    }
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Exp => {
                if let Ok(mut a) = a.write() {
                    a.grad += out.data * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::Pow => {
                if let (Ok(mut a), Ok(b)) = (a.write(), b.read()) {
                    let x = b.data;
                    a.grad += x * a.data.powf(x - 1.0) * out.grad;
                } else {
                    panic!("Failed to get reference to value");
                }
            }
            Op::None => {}
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Mul => "x",
            Op::Tanh => "tanh",
            Op::Relu => "relu",
            Op::Exp => "exp",
            Op::Pow => "pow",
            Op::None => "",
        }
    }
}

/// Unique identifier for [`Value`]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(usize);

impl Id {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Debug, Clone)]
pub struct Value_ {
    pub data: f64,
    prev: Vec<ValueRef>,
    pub op: Op,
    pub grad: f64,
    pub label: Option<String>,
    id: Id,
}

impl Value_ {
    fn new(data: f64) -> Self {
        Self {
            data,
            op: Op::None,
            prev: vec![],
            grad: 0.0,
            label: None,
            id: Id::new(),
        }
    }

    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    fn backward(&self) {
        match self.prev.len() {
            1 => {
                self.op.backward(&self, &self.prev[0], &self.prev[0]);
            }
            2 => {
                self.op.backward(&self, &self.prev[0], &self.prev[1]);
            }
            _ => {}
        }
    }
}

type ValueRef = Arc<RwLock<Value_>>;

#[derive(Debug, Clone)]
pub struct Value(ValueRef);

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Arc::new(RwLock::new(Value_::new(data))))
    }

    pub fn with_label(self, label: &str) -> Self {
        if let Ok(mut p) = self.0.write() {
            p.label = Some(label.to_owned());
        }
        self
    }

    fn new_from_op(data: f64, op: Op, prev: Vec<ValueRef>) -> Self {
        Self(Arc::new(RwLock::new(Value_ {
            data,
            op,
            prev,
            grad: 0.0,
            label: None,
            id: Id::new(),
        })))
    }

    pub fn data(&self) -> f64 {
        if let Ok(v) = self.0.read() {
            v.data
        } else {
            panic!("Failed to get reference to value");
        }
    }

    pub fn set_data(&self, value: f64) {
        if let Ok(mut p) = self.0.write() {
            p.data = value;
        }
    }

    pub fn grad(&self) -> f64 {
        self.0.read().unwrap().grad
    }

    pub fn step(&self, size: f64) {
        if let Ok(mut p) = self.0.write() {
            p.data -= size.abs() * p.grad;
        }
    }

    pub fn tanh(self) -> Self {
        Self::new_from_op(self.data().tanh(), Op::Tanh, vec![self.0.clone()])
    }

    pub fn relu(self) -> Self {
        let data = self.data();
        let relu = if data > 0.0 { data } else { 0.0 };
        Self::new_from_op(relu, Op::Relu, vec![self.0.clone()])
    }

    pub fn exp(self) -> Self {
        Self::new_from_op(self.data().exp(), Op::Exp, vec![self.0.clone()])
    }

    pub fn pow(self, exp: Value) -> Self {
        Self::new_from_op(
            self.data().powf(exp.data()),
            Op::Pow,
            vec![self.0.clone(), exp.0.clone()],
        )
    }

    /// Back-propagation to compute gradients
    pub fn backward(&self) {
        if let Ok(mut inner) = self.0.write() {
            inner.grad = 1.0;
        }

        let mut topo = Vec::new();
        let mut visited = Vec::new();
        build_topo(&mut topo, &mut visited, self.0.clone());

        for v in topo.iter().rev() {
            v.read().unwrap().backward();
        }
    }

    /// Get the topological order of the graph
    pub fn get_topo(&self) -> Vec<ValueRef> {
        let mut topo = Vec::new();
        let mut visited = Vec::new();
        build_topo(&mut topo, &mut visited, self.0.clone());
        topo.reverse();
        topo
    }

    pub fn zero_grad(&self) {
        if let Ok(mut v) = self.0.write() {
            v.grad = 0.0;
        }
    }

    #[cfg(feature = "layout")]
    pub fn trace(&self) -> layout::topo::layout::VisualGraph {
        use layout::core::base::Orientation;
        use layout::std_shapes::shapes::*;
        use layout::topo::layout::VisualGraph;

        let mut vg = VisualGraph::new(Orientation::LeftToRight);
        if let Ok(v) = self.0.read() {
            let record = visual::create_record(v.label.clone(), v.data, v.grad);
            let mut handle = visual::draw_shape(&mut vg, ShapeKind::Record(record));
            if v.op != Op::None {
                let op =
                    visual::draw_shape(&mut vg, ShapeKind::Circle(v.op.to_string().to_owned()));
                let arrow = Arrow::simple("");
                vg.add_edge(arrow, op, handle);
                handle = op;
            }

            let node = visual::Node {
                value: self.0.clone(),
                handle,
            };
            visual::build_vg(&mut vg, node, &mut Vec::new(), &mut Vec::new());
        }
        vg
    }
}

#[cfg(feature = "layout")]
mod visual {
    use super::*;
    use layout::adt::dag::NodeHandle;
    use layout::core::{base::Orientation, geometry::Point, style::*};
    use layout::std_shapes::shapes::*;
    use layout::topo::layout::VisualGraph;

    pub fn draw_shape(vg: &mut VisualGraph, shape: ShapeKind) -> NodeHandle {
        let style = StyleAttr::simple();
        let size = match shape {
            ShapeKind::Box(_) => Point::new(100.0, 50.0),
            ShapeKind::Record(_) => Point::new(200.0, 50.0),
            _ => Point::new(50.0, 50.0),
        };
        let element = Element::create(shape, style, Orientation::LeftToRight, size);
        vg.add_node(element)
    }

    pub fn create_record(label: Option<String>, data: f64, grad: f64) -> RecordDef {
        let data = format!("{:.3}", data);
        let data = RecordDef::new_text(&data);
        let grad = format!("{:.3}", grad);
        let grad = RecordDef::new_text(&grad);
        let mut array = vec![data, grad];
        if let Some(label) = label {
            let label = RecordDef::new_text(&label);
            array.insert(0, label);
        }
        let record = RecordDef::Array(array);
        record
    }

    #[derive(Clone)]
    pub struct Node {
        pub value: ValueRef,
        pub handle: NodeHandle,
    }

    /// Build nodes and edges to visualize the graph
    pub fn build_vg(
        vg: &mut VisualGraph,
        node: Node,
        visited_hdls: &mut Vec<NodeHandle>,
        visited_ids: &mut Vec<Id>,
    ) {
        if let Ok(v) = node.value.read() {
            for child in v.prev.iter() {
                if let Ok(c) = child.read() {
                    match visited_ids.binary_search(&c.id) {
                        Ok(pos) => {
                            // Node already added, just add edge
                            let arrow = Arrow::simple("");
                            vg.add_edge(arrow, visited_hdls[pos], node.handle);
                        }
                        Err(pos) => {
                            // Node has not been added yet, create it
                            let record = create_record(c.label.clone(), c.data, c.grad);
                            let handle = draw_shape(vg, ShapeKind::Record(record));
                            let arrow = Arrow::simple("");
                            vg.add_edge(arrow, handle, node.handle);
                            if c.op != Op::None {
                                let op =
                                    draw_shape(vg, ShapeKind::Circle(c.op.to_string().to_owned()));
                                let arrow = Arrow::simple("");
                                vg.add_edge(arrow, op, handle);
                                // handle = op;

                                visited_hdls.insert(pos, handle);
                                visited_ids.insert(pos, c.id);

                                let n = Node {
                                    value: child.clone(),
                                    handle: op,
                                };
                                build_vg(vg, n, visited_hdls, visited_ids);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Build the topological order of the graph, used for computing backward gradients.
fn build_topo(topo: &mut Vec<ValueRef>, visited_ids: &mut Vec<Id>, value: ValueRef) {
    if let Ok(v) = value.read() {
        if v.op != Op::None {
            match visited_ids.binary_search(&v.id) {
                Ok(_) => {}
                Err(pos) => {
                    visited_ids.insert(pos, v.id);
                    for child in v.prev.iter() {
                        build_topo(topo, visited_ids, child.clone());
                    }
                    topo.push(value.clone());
                }
            }
        }
    }
}

impl Into<Value> for f64 {
    fn into(self) -> Value {
        Value::new(self)
    }
}

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        Self::new_from_op(
            self.data() + rhs.data(),
            Op::Add,
            vec![self.0.clone(), rhs.0.clone()],
        )
    }
}

impl std::ops::Add<f64> for Value {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        let rhs = Value::new(rhs);
        self + rhs
    }
}

impl std::ops::Neg for Value {
    type Output = Self;

    fn neg(self) -> Self {
        self * -1.0
    }
}

impl std::ops::Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl std::ops::Sub<f64> for Value {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        let rhs = Value::new(rhs);
        self - rhs
    }
}

impl std::ops::Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self * (rhs.pow(Value::new(-1.0)))
    }
}

impl std::ops::Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new_from_op(
            self.data() * rhs.data(),
            Op::Mul,
            vec![self.0.clone(), rhs.0.clone()],
        )
    }
}

impl std::ops::Mul<f64> for Value {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        let rhs = Value::new(rhs);
        self * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::float_eq;

    #[test]
    fn test_backward() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2.clone() + b;
        let o = n.clone().tanh();

        o.backward();

        assert!(float_eq(0.7071, o.data()));
        assert!(float_eq(0.5, n.grad()));
        assert!(float_eq(0.5, x2.grad()));
        assert!(float_eq(0.0, w2.grad()));
        assert!(float_eq(1.0, w1.grad()));
        assert!(float_eq(-1.5, x1.grad()));
    }

    #[test]
    fn test_backward_exp() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);
        let b = Value::new(6.8813735870195432);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2.clone() + b;
        let e = (n.clone() * 2.0).exp();
        let o = (e.clone() - 1.0) / (e + 1.0);

        o.backward();

        assert!(float_eq(0.7071, o.data()));
        assert!(float_eq(0.5, n.grad()));
        assert!(float_eq(0.5, x2.grad()));
        assert!(float_eq(0.0, w2.grad()));
        assert!(float_eq(1.0, w1.grad()));
        assert!(float_eq(-1.5, x1.grad()));
    }

    #[test]
    fn test_exp() {
        let a = Value::new(0.6931471805599453);
        let b = a.clone().exp();
        let c = (b.clone() - 1.0) / (b.clone() + 1.0);
        c.backward();
        assert!(float_eq(1.0 / 3.0, c.data()));
        assert!(float_eq(2.0 / 3.0 - 2.0 / 9.0, a.grad()));
    }

    #[test]
    fn test_gradient_accumulation() {
        let a = Value::new(3.0);
        let b = a.clone() + a.clone();
        b.backward();
        assert!(float_eq(2.0, a.grad()));

        let a = Value::new(-2.0);
        let b = Value::new(3.0);
        let d = a.clone() * b.clone();
        let e = a.clone() + b.clone();
        let f = d * e;
        f.backward();
        assert!(float_eq(-3.0, a.grad()));
        assert!(float_eq(-8.0, b.grad()));
    }

    #[test]
    fn test_ops() {
        let a = Value::new(2.0);
        let b = Value::new(4.0);
        let c = a.clone() / b.clone();
        c.backward();
        assert!(float_eq(0.5, c.data()));
        assert!(float_eq(0.25, a.grad()));
        assert!(float_eq(-0.125, b.grad()));

        let a = Value::new(2.0);
        let b = Value::new(4.0);
        let c = a.clone() - b.clone();
        c.backward();
        assert!(float_eq(-2.0, c.data()));
        assert!(float_eq(1.0, a.grad()));
        assert!(float_eq(-1.0, b.grad()));

        let a = Value::new(2.0);
        let b = a.clone().exp();
        b.backward();
        assert!(float_eq(7.38906, b.data()));
        assert!(float_eq(7.38906, a.grad()));
    }

    #[test]
    #[cfg(feature = "layout")]
    fn test_graph_vis() {
        use layout::backends::svg::SVGWriter;
        use layout::core::utils::save_to_file;

        let x1 = Value::new(2.0).with_label("x1");
        let x2 = Value::new(0.0).with_label("x2");
        let w1 = Value::new(-3.0).with_label("w1");
        let w2 = Value::new(1.0).with_label("w2");
        let b = Value::new(6.8813735870195432).with_label("b");

        let x1w1 = (x1.clone() * w1.clone()).with_label("x1w1");
        let x2w2 = (x2.clone() * w2.clone()).with_label("x2w2");
        let x1w1x2w2 = (x1w1.clone() + x2w2.clone()).with_label("x1w1x2w2");
        let n = (x1w1x2w2.clone() + b).with_label("n");
        let e = ((n.clone() * 2.0).exp()).with_label("e");
        let o = ((e.clone() - 1.0) / (e + 1.0)).with_label("o");
        o.backward();

        let mut vg = o.trace();
        let mut svg = SVGWriter::new();
        vg.do_it(false, false, false, &mut svg);
        let _ = save_to_file("graph.svg", &svg.finalize());
    }
}
