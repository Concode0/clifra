import torch

from clifra import make_algebra
from clifra.core import LaneStorage, format_multivector

algebra = make_algebra(30, 30, 2, device="cpu", dtype=torch.float64)

vector_layout = algebra.layout((1,))

left = torch.rand(1, vector_layout.dim)
right = torch.rand(1, vector_layout.dim)

result, result_layout = algebra.wedge(
    left,
    right,
    left_layout=vector_layout,
    right_layout=vector_layout,
    output_storage=LaneStorage.COMPACT,
    return_layout=True,
)

pseudoscalar_result, pseudoscalar_layout = algebra.pseudoscalar_product(
    result,
    input_layout=result_layout,
    return_layout=True,
)

print(left)
print(right)

print(result_layout)

print(format_multivector(algebra, result, layout=result_layout))

print(pseudoscalar_layout)

print(format_multivector(algebra, pseudoscalar_result, layout=pseudoscalar_layout))
