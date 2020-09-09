#include <epidemiology/utils/logging.h>
#include <gtest/gtest.h>

#if HAVE_EPI_IO
#include <epidemiology_io/io.h>
#endif

int main(int argc, char** argv)
{
    epi::set_log_level(epi::LogLevel::warn);
    ::testing::InitGoogleTest(&argc, argv);
    int retval = RUN_ALL_TESTS();

#if HAVE_EPI_IO
    epi::io_cleanup();
#endif
    return retval;
}
