// Why the heck does factorial take a float?
// This is to take advantage of the fact that std::f64::MAX >>> std::u64::MAX
fn factorial_unsafe(num: f64) -> f64 {
    if num == 0. || num == 1. {
        return 1.;
    } else {
        return num * factorial_unsafe(num - 1.);
    }
}

pub fn factorial(num: f64) -> Result<f64, &'static str> {
    if num.fract() != 0. || num < 0. {
        return Err("Number must be non-negative with no fractional component!");
    } else if num > 170. {
        return Ok(std::f64::INFINITY);
    } else {
        return Ok(factorial_unsafe(num));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0.), Ok(1.));
        assert_eq!(factorial(1.), Ok(1.));
        assert_eq!(factorial(2.), Ok(2.));
        assert_eq!(factorial(3.), Ok(6.));
        assert_eq!(factorial(170.), Ok(7.257415615307994e306));
        assert_eq!(factorial(171.), Ok(std::f64::INFINITY));

        match factorial(1.1) {
            Err(_) => {}
            _ => panic!("Shouldn't be able to do factorial on number with fractional component!"),
        }

        match factorial(-1.) {
            Err(_) => {}
            _ => panic!("Shouldn't be able to do factorial on negative number!"),
        }
    }
}
