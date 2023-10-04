#include <FlyTorchTests.h>

bool unit_test_main()
{
	return true;
}

int main(int argc, char* argv[])
{
	return boost::unit_test::unit_test_main(unit_test_main, argc, argv);
}
