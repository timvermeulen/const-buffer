use super::*;

#[test]
fn read() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        for i in 0..10 {
            buffer.as_mut_ptr().add(i).write(i * 10);
        }

        for i in 0..10 {
            assert_eq!(buffer.read(i), i * 10);
        }
    }
}

#[test]
fn write() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        for i in 0..10 {
            *buffer.write(i, i * 10) += 5;
        }

        for i in 0..10 {
            assert_eq!(buffer.as_ptr().add(i).read(), i * 10 + 5);
        }
    }
}

#[test]
fn get() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        buffer.write(0, 0);
        buffer.write(1, 10);
        buffer.write(2, 20);
        buffer.write(8, 80);
        buffer.write(9, 90);

        assert_eq!(buffer.get(0), &0);
        assert_eq!(buffer.get(1), &10);
        assert_eq!(buffer.get(2), &20);
        assert_eq!(buffer.get(8), &80);
        assert_eq!(buffer.get(9), &90);

        assert_eq!(buffer.get(..3), &[0, 10, 20]);
        assert_eq!(buffer.get(..=1), &[0, 10]);
        assert_eq!(buffer.get(1..=1), &[10]);
        assert_eq!(buffer.get(1..3), &[10, 20]);
        assert_eq!(buffer.get(8..), &[80, 90]);
    }
}

#[test]
fn get_mut() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        buffer.write(0, 0);
        buffer.write(1, 10);
        buffer.write(2, 20);
        buffer.write(8, 80);
        buffer.write(9, 90);

        let x = buffer.get_mut(0);
        assert_eq!(*x, 0);
        *x += 5;
        assert_eq!(*x, 5);

        assert_eq!(buffer.get_mut(..3), &mut [5, 10, 20]);
        assert_eq!(buffer.get_mut(..=1), &mut [5, 10]);
        assert_eq!(buffer.get_mut(1..=1), &mut [10]);
        assert_eq!(buffer.get_mut(1..3), &mut [10, 20]);
        assert_eq!(buffer.get_mut(8..), &mut [80, 90]);

        let x = buffer.get_mut(..3);
        x.reverse();
        assert_eq!(x, &mut [20, 10, 5]);
        x.rotate_left(1);
        assert_eq!(x, &mut [10, 5, 20]);

        assert_eq!(buffer.read(0), 10);
        assert_eq!(buffer.read(1), 5);
        assert_eq!(buffer.read(2), 20);
    }
}

#[test]
fn swap() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        buffer.write(3, 30);
        buffer.write(5, 50);

        buffer.swap(3, 3);
        buffer.swap(3, 5);
        buffer.swap(5, 5);

        assert_eq!(buffer.read(3), 50);
        assert_eq!(buffer.read(5), 30);
    }
}

#[test]
fn swap_nonoverlapping() {
    let mut buffer = ConstBuffer::<usize, 10>::new();

    unsafe {
        buffer.write(3, 30);
        buffer.write(5, 50);

        buffer.swap_nonoverlapping(3, 5);

        assert_eq!(buffer.read(3), 50);
        assert_eq!(buffer.read(5), 30);
    }
}

#[test]
fn resize() {
    let mut buffer = ConstBuffer::<u32, 10>::new();

    unsafe {
        buffer.write(3, 30);
        buffer.write(7, 70);
    }

    let mut small: ConstBuffer<u32, 5> = buffer.resize();
    let large: ConstBuffer<u32, 15> = buffer.resize();

    unsafe {
        assert_eq!(small.read(3), 30);
        assert_eq!(large.read(3), 30);
        assert_eq!(large.read(7), 70);

        *small.get_mut(3) += 1;
        assert_eq!(small.read(3), 31);
        assert_eq!(large.read(3), 30);

        *buffer.get_mut(7) -= 1;
        assert_eq!(buffer.read(7), 69);
        assert_eq!(large.read(7), 70);
    }
}

#[test]
fn copy_within() {
    let mut buffer = ConstBuffer::<u32, 10>::new();

    unsafe {
        buffer.write(2, 20);
        buffer.write(5, 50);
        buffer.write(6, 60);
        buffer.write(7, 70);

        buffer.copy_within(4.., 1);
        assert_eq!(buffer.get(2..5), &[50, 60, 70]);
        assert_eq!(buffer.read(7), 70);
    }
}

#[test]
fn copy_within_nonoverlapping() {
    let mut buffer = ConstBuffer::<u32, 10>::new();

    unsafe {
        buffer.write(2, 20);
        buffer.write(5, 50);
        buffer.write(6, 60);
        buffer.write(7, 70);

        buffer.copy_within(4..7, 1);
        assert_eq!(buffer.get(2..4), &[50, 60]);
        assert_eq!(buffer.get(5..8), &[50, 60, 70]);
    }
}

#[test]
fn copy_from_slice() {
    let mut vec = vec![1, 2, 3];
    let mut buffer = ConstBuffer::<u32, 10>::new();

    unsafe {
        buffer.copy_from_slice(3, &vec);
        vec.reverse();
        buffer.copy_from_slice(6, &vec);
        assert_eq!(buffer.get(3..9), &[1, 2, 3, 3, 2, 1]);
    }
}

#[test]
fn clone_from_slice() {
    let vec = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let mut buffer = ConstBuffer::<std::vec::Vec<u32>, 10>::new();

    unsafe {
        buffer.clone_from_slice(3, &vec);
        let mut x = buffer.read(4);
        x.reverse();
        assert_eq!(x, &[6, 5, 4]);
    }

    assert_eq!(vec[1], &[4, 5, 6]);
}
