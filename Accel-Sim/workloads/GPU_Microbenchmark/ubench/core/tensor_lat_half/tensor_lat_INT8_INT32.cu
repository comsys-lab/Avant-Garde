#include "tensor_lat_half.h"

int main() {

  intilizeDeviceProp(0);

  if (deviceProp.major < 6) // tesnore unit was added since Volta
    return 1;

  std::cout << "\nint8_t operand, int accumalte:\n";
  tensor_lat<int8_t,int>();

  return 1;
}
