import numpy as np

def einsum(contraction_pattern, *arrays):
  subscripts, target_subscript = _process_einsum_subscripts(contraction_pattern)
  array1 = arrays[0]
  subscript1 = subscripts[0]
  for array2, subscript2 in zip(arrays[1:], subscripts[1:]):
    contracted_axes = _get_contracted_axes(subscript1, subscript2)
    array1 = np.tensordot(array1, array2, axes = contracted_axes)
    subscript1 = _get_subscript_after_contraction(subscript1, subscript2)
  if target_subscript is '':
    target_subscript = subscript1
  axis_order = tuple(subscript1.index(index) for index in target_subscript)
  return array1.transpose(*axis_order)
    
  
def _process_einsum_subscripts(contraction_pattern):
  target_subscript = ''
  if '->' in contraction_pattern:
    contraction_pattern, target_subscript = contraction_pattern.split('->')
  subscripts = contraction_pattern.split(',')
  return subscripts, target_subscript

def _get_contracted_axes(subscript1, subscript2):
  array1_axes = [subscript1.index(index) for index in subscript1
                 if index in subscript2]
  array2_axes = [subscript2.index(index) for index in subscript1
                 if index in subscript2]
  return (array1_axes, array2_axes)

def _get_subscript_after_contraction(subscript1, subscript2):
  return ''.join(index for index in subscript1 + subscript2
                 if not (index in subscript1 and index in subscript2))

if __name__ == "__main__":
  a = np.random.rand(5, 6)
  b = np.random.rand(6, 7)
  c = np.random.rand(7, 8)
  d_ref = a.dot(b.dot(c))
  d = einsum("ij,jk,kl", a, b, c)
  print(np.allclose(d, d_ref))
  
