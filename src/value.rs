use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Tanh,
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

    fn to_string(&self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Mul => "x",
            Op::Tanh => "tanh",
            Op::Exp => "exp",
            Op::Pow => "pow",
            Op::None => "",
        }
    }
}

/// Unique identifier for [`Value`]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
struct Value_ {
    data: f64,
    prev: Vec<ValueRef>,
    op: Op,
    grad: f64,
    id: Id,
}

impl Value_ {
    fn new(data: f64) -> Self {
        Self {
            data,
            op: Op::None,
            prev: vec![],
            grad: 0.0,
            id: Id::new(),
        }
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

    fn new_from_op(data: f64, op: Op, prev: Vec<ValueRef>) -> Self {
        Self(Arc::new(RwLock::new(Value_ {
            data,
            op,
            prev,
            grad: 0.0,
            id: Id::new(),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.read().unwrap().data
    }

    pub fn grad(&self) -> f64 {
        self.0.read().unwrap().grad
    }

    pub fn step(&self, size: f64) {
        if let Ok(mut v) = self.0.write() {
            v.data += -size.abs() * v.grad;
        }
    }

    pub fn tanh(self) -> Self {
        Self::new_from_op(self.data().tanh(), Op::Tanh, vec![self.0.clone()])
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

    pub fn backward(&self) {
        if let Ok(mut inner) = self.0.write() {
            inner.grad = 1.0;
        }

        let mut topo = Vec::new();
        let mut visited = Vec::new();
        build_topo(&mut topo, &mut visited, self.0.clone());

        let mut cnt = 0;
        for v in topo.iter().rev() {
            if let Ok(v) = v.read() {
                if v.op == Op::Tanh {
                    cnt = cnt + 1;
                }
            }
        }
        // println!("# tanh = {}", cnt);

        // println!("{}", topo.len());
        for v in topo.iter().rev() {
            v.read().unwrap().backward();
            // if let Ok(v) = v.read() {
            //     println!("{:>5}|{:7.3}|{:7.3}", v.op.to_string(), v.data, v.grad);
            // }
        }
    }
}

fn build_topo(topo: &mut Vec<ValueRef>, visited_ids: &mut Vec<Id>, value: ValueRef) {
    if let Ok(v) = value.read() {
        if !visited_ids.contains(&v.id) && v.op != Op::None {
            visited_ids.push(v.id);
            for child in v.prev.iter() {
                build_topo(topo, visited_ids, child.clone());
            }
            topo.push(value.clone());
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

    fn float_eq(exp: f64, act: f64) -> bool {
        if exp < std::f64::EPSILON {
            if act < std::f64::EPSILON {
                true
            } else {
                false
            }
        } else {
            let tol = exp * 1.0e-4;
            (exp - act).abs() < tol
        }
    }

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
}
