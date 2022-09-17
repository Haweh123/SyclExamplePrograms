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
 * // Do a USM malloc_device
 * auto ptr = sycl::malloc_device<T>(n, q);
 *
 * // Do a USM memcpy
 * q.memcpy(dst_ptr, src_ptr, sizeof(T)*n);
 *
 * // Wait on a queue
 * q.wait();
 *
 * // Submit work to the queue
 * q.submit([&](sycl::handler &cgh) {
 *   // COMMAND GROUP
 * });
 *
 *
 * // Within the command group you can
 * //    1. Declare an accessor to a buffer
 *          auto read_write_acc = sycl::accessor{buf, cgh};
 *          auto read_acc = sycl::accessor{buf, cgh, sycl::read_only};
 *          auto write_acc = sycl::accessor{buf, cgh, sycl::write_only};
 *          auto no_init_acc = sycl::accessor{buf, cgh, sycl::no_init};
 * //    2. Enqueue a parallel for:
 *              cgh.parallel_for<class mykernel>(sycl::range{n}, 
 *                    [=](sycl::id<1> i) { // Do something });
 *
 *
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class scalar_add_usm;
class scalar_add_buff_acc;

class usm_selector : public sycl::device_selector {
 public:
  int operator()(const sycl::device& dev) const {
    if (dev.has(sycl::aspect::usm_device_allocations)) {
      if (dev.has(sycl::aspect::gpu)) return 2;
      return 1;
    }
    return -1;
  }
};
TEST_CASE("synchronization_usm", "synchronization_source") {
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

    auto usmQueue = sycl::queue{usm_selector{}, asyncHandler};

    auto devicePtrA = sycl::malloc_device<float>(dataSize, usmQueue);
    auto devicePtrB = sycl::malloc_device<float>(dataSize, usmQueue);
    auto devicePtrR = sycl::malloc_device<float>(dataSize, usmQueue);

    usmQueue.memcpy(devicePtrA, a,
                    sizeof(float) * dataSize)
        .wait();  // Synchronize
    usmQueue.memcpy(devicePtrB, b,
                    sizeof(float) * dataSize)
        .wait();  // Synchronize

    usmQueue
        .parallel_for<scalar_add_usm>(sycl::range{dataSize},
                                    [=](sycl::id<1> idx) {
                                      auto globalId = idx[0];
                                      devicePtrR[globalId] =
                                          devicePtrA[globalId] +
                                          devicePtrB[globalId];
                                    })
        .wait();  // Synchronize

    usmQueue.memcpy(r, devicePtrR,
                    sizeof(float) * dataSize)
        .wait();  // Synchronize and copy-back

    sycl::free(devicePtrA, usmQueue);
    sycl::free(devicePtrB, usmQueue);
    sycl::free(devicePtrR, usmQueue);

    usmQueue.throw_asynchronous();
  } catch (const sycl::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int i = 0; i < dataSize; ++i) {
    REQUIRE(r[i] == i * 2);
  }
}

TEST_CASE("synchronization_buffer_acc", "synchronization_source") {
  int a = 18, b = 24, r = 0;
  auto defaultQueue = sycl::queue{};

  {
    auto bufA = sycl::buffer{&a, sycl::range{1}};
    auto bufB = sycl::buffer{&b, sycl::range{1}};
    auto bufR = sycl::buffer{&r, sycl::range{1}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
          auto accB = sycl::accessor{bufB, cgh, sycl::read_only};
          auto accR = sycl::accessor{bufR, cgh, sycl::write_only};

          cgh.single_task<scalar_add_buff_acc>([=] { accR[0] = accA[0] + accB[0]; });
        });
  }

  REQUIRE(r == 42);
}
