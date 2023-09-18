pub fn float_eq(exp: f64, act: f64) -> bool {
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
