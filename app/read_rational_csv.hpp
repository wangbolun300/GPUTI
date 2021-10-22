#pragma once

#include <vector>
#include<array>
#include <gputi/CType.hpp>
#include <string>
namespace ccd {

std::vector<std::array<Scalar,3>> 
read_rational_csv(const std::string& inputFileName, std::vector<bool>& results);

} // namespace ccd