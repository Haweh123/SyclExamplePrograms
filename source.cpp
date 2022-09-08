/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#if __has_include("SYCL/sycl.hpp")
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class my_kernel;
TEST_CASE("hello_world", "hello_world_source") {

  // Print "Hello World!\n"
  std::cout << "Hello World!\n";

  sycl::queue GpuQueue;
  
  GpuQueue.submit([&] (sycl::handler &cph) {
  	sycl::stream os  = sycl::stream(1024,128,cph);
	cph.single_task<my_kernel>([=]() {
		os << "Hello from device\n";
	});
  }).wait();
  REQUIRE(true);
}
