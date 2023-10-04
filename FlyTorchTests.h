#pragma once

#include <FlyTorch.h>
using namespace FlyTorch;

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(FlyTorchTests);

BOOST_AUTO_TEST_CASE(test_vector)
{
	auto out = std::make_shared<Vector>(2);
}


BOOST_AUTO_TEST_CASE(test_tensor)
{
	//Tensor s1(1.0f);
	//Tensor v1(1.0f, 2.0f, 3.0f);

	//BOOST_CHECK_NO_THROW(auto s2 = std::get<std::shared_ptr<Scalar>>(s1));
	//BOOST_CHECK_NO_THROW(auto v2 = std::get<std::shared_ptr<Vector>>(v1));
}

BOOST_AUTO_TEST_CASE(test_tensor_add)
{
	//Tensor s1(2.0f);
	//Tensor s2(2.0f);
	//Tensor s3 = s1 + s2;
	//BOOST_CHECK(s3.data() == 4.0f);

	//Tensor v1(1.0f, 2.0f);
	//Tensor v2(3.0f, 4.0f);
	//Tensor v3 = v1 + v2;
	//BOOST_CHECK(v3.data(0) == 4.0f);
	//BOOST_CHECK(v3.data(1) == 6.0f);
}

BOOST_AUTO_TEST_CASE(test_mlp)
{
	std::vector<std::shared_ptr<Vector>> xs = {
		std::make_shared<Vector>(2.0f, .0f, 1.0f),
		std::make_shared<Vector>(3.0f, 1.0f, 0.5f),
		std::make_shared<Vector>(0.5f, .0f, 1.0f),
		std::make_shared<Vector>(1.0f, .0f, -1.0f),
	};

	auto ys = std::make_shared<Vector>(-1.0f, 1.0f, -1.0f, 1.0f);

	MLP mlp(3, { 4, 4, 1 });

	auto losses = std::make_shared<Vector>(xs.size());
	std::vector<std::shared_ptr<Vector>> ypred(xs.size());

	for (size_t i = 0; i < 100; i++) {

		for (size_t i = 0; i < xs.size(); i++) {
			ypred[i] = mlp(xs[i]);
			(*losses)[i] = ((*ypred[i])[0] - (*ys)[i]) ^ 2; // TODO: element-wise operator-()
		}

		// zero grads
		mlp.parameters([](std::shared_ptr<Scalar>& p) {
			p->grad = 0.0f; });

		auto loss = std::accumulate(std::next(losses->begin()), losses->end(), losses->front());
		loss->backward();

		// gradient descent
		mlp.parameters([](std::shared_ptr<Scalar>& p) {
			p->data -= 0.05f * p->grad; });

		if (i % 10 == 0) {
			std::cout << "iteration: " << i << " loss: " << loss->data << std::endl;
		}
	}

	std::cout << "ys:" << std::endl;
	for (auto& y : *ys) {
		std::cout << y->data << ", ";
	}
	std::cout << std::endl;

	std::cout << "ypred:" << std::endl;
	for (auto& pred : ypred) {
		for (auto& p : *pred) {
			std::cout << p->data << ", ";
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();
