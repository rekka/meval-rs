pub fn factorial(num: u64) -> u64 {
	if num == 0 ||
		num == 1 {
		return 1;
	} else {
		return num * factorial(num - 1);
	}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
      assert_eq!(factorial(0), 1);
      assert_eq!(factorial(1), 1);
      assert_eq!(factorial(2), 2);
      assert_eq!(factorial(3), 6);
      assert_eq!(factorial(4), 24);
    }
}