#pragma once

#include <vector>
#include <array>
// #include <gputi/CType.hpp>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <gputi/CType.cuh>

namespace ccd
{
	std::vector<std::array<ccd::Scalar, 3>>
	read_rational_csv(const std::string &inputFileName, std::vector<bool> &results);

} // namespace ccd