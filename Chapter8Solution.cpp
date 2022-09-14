/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <catch2/catch.hpp>
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif 

class usm_selector : public sycl::device_selector {
   // Overload operator() for sycl::device.
   public:
   int operator()(const sycl::device& dev) const override {
      if (dev.has(sycl::aspect::usm_device_allocations)) {
		return 1;
      }
      return -1;
   }
 };

class vector_add;
TEST_CASE("usm_vector_add", "usm_vector_add_source") {
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  try {
    auto asyncHandler = [&](sycl::exception_list exceptionList) {
      for (auto& e : exceptionList) {
        std::rethrow_exception(e);
      }
    };

    auto defaultQueue = sycl::queue{usm_selector{}, asyncHandler};
   
    auto devicePtrA = sycl::malloc_device<float>(dataSize,defaultQueue);
    auto devicePtrB = sycl::malloc_device<float>(dataSize,defaultQueue);
    auto devicePtrR = sycl::malloc_device<float>(dataSize,defaultQueue);

    defaultQueue.memcpy(devicePtrA, a, sizeof(float) * dataSize);
    defaultQueue.memcpy(devicePtrB, a, sizeof(float) * dataSize);
 
    defaultQueue.parallel_for<vector_add>(
		sycl::range{dataSize},[=](sycl::id<1> idx) {
		auto globalId = idx[0];
		devicePtrR[globalId] = devicePtrA[globalId] + devicePtrB[globalId];
	 }).wait();    

    defaultQueue.memcpy(r, devicePtrR, sizeof(float) * dataSize).wait();
    sycl::free(devicePtrA, defaultQueue);
    sycl::free(devicePtrB, defaultQueue);
    sycl::free(devicePtrR, defaultQueue);

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == i * 2);
  }
}
