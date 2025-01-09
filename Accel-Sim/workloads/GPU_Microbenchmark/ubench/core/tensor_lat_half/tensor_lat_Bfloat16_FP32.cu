#include "tensor_lat_half.h"

int main() {

  intilizeDeviceProp(0);

  if (deviceProp.major < 6) // tesnore unit was added since Volta
    return 1;

  std::cout << "\n__nv_bfloat16 operand, float accumalte:\n";
  tensor_lat<__nv_bfloat16, float>();

  return 1;
}
