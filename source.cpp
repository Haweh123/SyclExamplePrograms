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

class add;
TEST_CASE("scalar_add", "scalar_add_source") {

  int a = 18, b = 24, r = 0;

  auto AdditionQueue = sycl::queue{};

  {
  auto buffA = sycl::buffer(&a,sycl::range{1});
  auto buffB = sycl::buffer(&b,sycl::range{1});
  auto buffAnswer = sycl::buffer(&r,sycl::range{1});

  
  AdditionQueue.submit([&](sycl::handler &cgh){
	  auto readAccA = sycl::accessor{buffA,cgh,sycl::read_only};
	  auto readAccB = sycl::accessor{buffB,cgh,sycl::read_only};
	  auto readAccAnswer = sycl::accessor{buffAnswer,cgh,sycl::write_only};
	  cgh.single_task<add>([=] {
		readAccAnswer[0] = readAccA[0] + readAccB[0];
	 });
    }).wait();
 }

  REQUIRE(r == 42);
}
