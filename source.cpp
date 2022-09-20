/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Declare a buffer pointing to ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}};
 *
 * // Do a USM memcpy
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 * // Do a USM memcpy with dependent events
 * auto event = q.memcpy(dst_ptr, src_ptr, sizeof(T)*n, {event1, event2});
 *
 * // Wait on an event
 * event.wait();
 *
 * // Wait on a queue
 * q.wait();
 *
 * // Submit work to the queue
 * auto event = q.submit([&](sycl::handler &cgh) {
 *   // COMMAND GROUP
 * });
 *
 *
 * // Within the command group you can
 * //    1. Declare an accessor to a buffer
 *          auto read_only_write_only_acc = sycl::accessor{buf, cgh};
 *          auto read_only_acc = sycl::accessor{buf, cgh, sycl::read_only_only};
 *          auto write_only_acc = sycl::accessor{buf, cgh, sycl::write_only_only};
 *          auto no_init_acc = sycl::accessor{buf, cgh, sycl::no_init};
 * //    2. Enqueue a parallel for:
 * //             i: Without dependent events
 *                    cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    [=](sycl::id<1> i) { // Do something });
 * //             ii: With dependent events
 *                    cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    {event1, event2}, [=](sycl::id<1> i) { 
 *                        // Do something
 *                      });
 *
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class a_1;
class b_1;
class c_1;
class d_1;
TEST_CASE("managing_dependencies", "managing_dependencies_source") {
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], c[dataSize], out[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    c[i] = static_cast<float>(i);
    out[i] = 0.0f;
  }

  try {
    auto asyncHandler = [&](sycl::exception_list exceptionList) {
      for (auto& e : exceptionList) {
        std::rethrow_exception(e);
      }
    };

    auto defaultQueue = sycl::queue{sycl::default_selector{}, asyncHandler};
    auto buffInA = sycl::buffer(a,sycl::range{dataSize});
    auto buffInB = sycl::buffer(b,sycl::range{dataSize});
    auto buffInC = sycl::buffer(c,sycl::range{dataSize});
    auto buffOut = sycl::buffer(out,sycl::range{dataSize});

    defaultQueue.submit([&](sycl::handler &cgh){
	  sycl::accessor accAIn{buffInA,cgh,sycl::read_write};
	  cgh.parallel_for<a_1>(
		sycl::range{dataSize},[=](sycl::id<1> idx) {
		   accAIn[idx] =  accAIn[idx] * 2;
	 });
    }).wait();

   defaultQueue.submit([&](sycl::handler &cgh){
	  sycl::accessor accIn{buffInA,cgh,sycl::read_only};
	  sycl::accessor accOut{buffOut,cgh,sycl::write_only};
	  cgh.parallel_for<b_1>(
		sycl::range{dataSize},[=](sycl::id<1> idx) {
		   accOut[idx] += accIn[idx];
	 });
    }).wait();

   defaultQueue.submit([&](sycl::handler &cgh){
	  sycl::accessor accIn{buffInA,cgh,sycl::read_only};
	  sycl::accessor accOut{buffInC,cgh,sycl::write_only};
	  cgh.parallel_for<c_1>(
		sycl::range{dataSize},[=](sycl::id<1> idx) {
		   accOut[idx] -= accIn[idx];
	 });
    }).wait();

   defaultQueue.submit([&](sycl::handler &cgh){
	  sycl::accessor accBIn{buffInB,cgh,sycl::read_only};
	  sycl::accessor accCIn{buffInC,cgh,sycl::read_only};
	  sycl::accessor accOut{buffInC,cgh,sycl::write_only};
	  cgh.parallel_for<d_1>(
		sycl::range{dataSize},[=](sycl::id<1> idx) {
		   accOut[idx] =  accBIn[idx]+ accCIn[idx];
	 });
    }).wait();

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(out[i] == i * 2.0f);
  }
}
