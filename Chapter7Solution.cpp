/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#define CATCH_CONFIG_MAIN
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
TEST_CASE("usm_selector", "usm_selector_source") {

  // Task: create a queue to a device which supports USM allocations
  try {
    auto asyncHandler = [&](sycl::exception_list exceptionList) {
      for (auto& e : exceptionList) {
        std::rethrow_exception(e);
      }
    };

    auto defaultQueue = sycl::queue{usm_selector{}, asyncHandler};
    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  REQUIRE(true);
}
