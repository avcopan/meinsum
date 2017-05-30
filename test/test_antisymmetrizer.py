import numpy as np
import os

from tensorutils.antisymmetrizer import get_antisymmetrizer_product as A

test_dir_path = os.path.dirname(os.path.realpath(__file__))
array_path_template = os.path.join(test_dir_path, "random_arrays", "{:s}.npy")


def test__composition_1():
    array1 = np.load(array_path_template.format("15x15"))
    array2 = A("0") * array1

    assert (np.allclose(array1, array2))


def test__composition_1_1():
    array1 = np.load(array_path_template.format("15x15"))

    array2 = A("0/1") * array1
    array3 = array1 - array1.transpose()

    assert (np.allclose(array2, array3))


def test__composition_1_2():
    array1 = np.load(array_path_template.format("15x15x15"))
    array2 = A("1/2") * array1

    array3 = A("0/1,2") * array2
    array4 = array2 - array2.transpose((1, 0, 2)) - array2.transpose((2, 1, 0))
    assert (np.allclose(array3, array4))


def test__composition_2_1():
    array1 = np.load(array_path_template.format("15x15x15"))
    array2 = A("0/1") * array1

    array3 = A("0,1/2") * array2
    array4 = array2 - array2.transpose((2, 1, 0)) - array2.transpose((0, 2, 1))
    assert (np.allclose(array3, array4))


def test__composition_1_1_1():
    array1 = np.load(array_path_template.format("15x15x15"))

    array2 = A("0/1/2") * array1
    array3 = (array1
              - array1.transpose((0, 2, 1))
              - array1.transpose((1, 0, 2))
              + array1.transpose((1, 2, 0))
              + array1.transpose((2, 0, 1))
              - array1.transpose((2, 1, 0)))
    assert (np.allclose(array2, array3))


def test__composition_1_3():
    array1 = np.load(array_path_template.format("15x15x15x15"))
    array2 = A("1/2/3") * array1

    array3 = A("0/1,2,3") * array2
    array4 = (array2
              - array2.transpose((1, 0, 2, 3))
              - array2.transpose((2, 1, 0, 3))
              - array2.transpose((3, 1, 2, 0)))
    assert (np.allclose(array3, array4))


def test__composition_2_2():
    array1 = np.load(array_path_template.format("15x15x15x15"))
    array2 = A("0/1|2/3") * array1

    array3 = A("0,1/2,3") * array2
    array4 = (array2
              - array2.transpose((2, 1, 0, 3))
              - array2.transpose((3, 1, 2, 0))
              - array2.transpose((0, 2, 1, 3))
              - array2.transpose((0, 3, 2, 1))
              + array2.transpose((2, 3, 0, 1)))
    assert (np.allclose(array3, array4))


def test__composition_3_1():
    array1 = np.load(array_path_template.format("15x15x15x15"))
    array2 = A("0/1/2") * array1

    array3 = A("0,1,2/3") * array2
    array4 = (array2
              - array2.transpose((3, 1, 2, 0))
              - array2.transpose((0, 3, 2, 1))
              - array2.transpose((0, 1, 3, 2)))

    assert (np.allclose(array3, array4))


def test__composition_1_2_1():
    array1 = np.load(array_path_template.format("15x15x15x15"))
    array2 = A("1/2") * array1

    array3 = A("0/1,2/3") * array2
    array4 = (array2
              - array2.transpose((1, 0, 2, 3))
              - array2.transpose((2, 1, 0, 3))
              - array2.transpose((3, 1, 2, 0))
              - array2.transpose((0, 3, 2, 1))
              - array2.transpose((0, 1, 3, 2))
              + array2.transpose((1, 0, 3, 2))
              + array2.transpose((2, 3, 0, 1))
              + array2.transpose((1, 3, 2, 0))
              + array2.transpose((2, 1, 3, 0))
              + array2.transpose((3, 0, 2, 1))
              + array2.transpose((3, 1, 0, 2)))
    assert (np.allclose(array3, array4))


def test__expression_01():
    array1 = np.load(array_path_template.format("15x15x15x15"))

    array2 = 0.25 * A("0/1|2/3") * array1

    array3 = 0.25 * (array1
                     - array1.transpose((1, 0, 2, 3))
                     - array1.transpose((0, 1, 3, 2))
                     + array1.transpose((1, 0, 3, 2)))

    assert (np.allclose(array2, array3))


def test__expression_02():
    array1 = np.load(array_path_template.format("15x15x15x15"))

    array2 = (0.25 * A("0/1")) * A("2/3") * array1

    array3 = 0.25 * (array1
                     - array1.transpose((1, 0, 2, 3))
                     - array1.transpose((0, 1, 3, 2))
                     + array1.transpose((1, 0, 3, 2)))

    assert (np.allclose(array2, array3))


def test__expression_03():
    array1 = np.load(array_path_template.format("15x15x15x15"))

    array2 = A("0/1") * (A("2/3") * 0.25) * array1

    array3 = 0.25 * (array1
                     - array1.transpose((1, 0, 2, 3))
                     - array1.transpose((0, 1, 3, 2))
                     + array1.transpose((1, 0, 3, 2)))

    assert (np.allclose(array2, array3))


if __name__ == "__main__":
    test__composition_1()
    test__composition_1_1()
    test__composition_1_2()
    test__composition_2_1()
    test__composition_1_1_1()
    test__composition_1_3()
    test__composition_2_2()
    test__composition_3_1()
    test__composition_1_2_1()
    test__expression_01()
    test__expression_02()
    test__expression_03()
