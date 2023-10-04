#pragma once

#include <set>
#include <random>
#include <ranges>
#include <format>
#include <memory>
#include <variant>
#include <functional>
#include <unordered_set>

namespace FlyTorch {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distribution(-1, 1);

	struct Node {
		Node(std::string op) : op(std::move(op)) {}
		std::string op{ "" };
		std::set<std::shared_ptr<Node>> parents{};
		std::function<void()> backward{ []() {} };
	};

	struct Scalar {
		Scalar(float data) :
			data(data)
		{}

		std::string format()
		{
			return std::format("Scalar data={0:.5f}, grad={1:.5f}", this->data, this->grad);
		}

		void backward()
		{
			std::vector<std::shared_ptr<Node>> topo;
			std::unordered_set<std::shared_ptr<Node>> visited;

			std::function<void(std::shared_ptr<Node>)> build_topo =
				[&](std::shared_ptr<Node> node)
				{
					if (node && !visited.contains(node)) {
						visited.insert(node);
						for (const auto& parent : node->parents) {
							build_topo(parent);
						}
						topo.emplace_back(node);
					}
				};

			build_topo(this->node);

			this->grad = 1.0f;
			for (auto& v : std::ranges::reverse_view(topo)) {
				v->backward();
			}
		}

		float data{ 0 };
		float grad{ 0 };
		std::shared_ptr<Node> node{ nullptr };
	};

	std::shared_ptr<Scalar> tanh(const std::shared_ptr<Scalar>& self)
	{
		auto out = std::make_shared<Scalar>((std::exp(2 * self->data) - 1) / (std::exp(2 * self->data) + 1));

		out->node = std::make_shared<Node>("tanh");
		out->node->parents = { self->node };
		out->node->backward = [self, out = out.get()]()
			{
				self->grad += (1.0f - out->data * out->data) * out->grad;
			};

		return out;
	}

	std::shared_ptr<Scalar> operator+(const std::shared_ptr<Scalar>& a, const std::shared_ptr<Scalar>& b)
	{
		auto c = std::make_shared<Scalar>(a->data + b->data);

		c->node = std::make_shared<Node>("+");
		c->node->parents = { a->node, b->node };
		c->node->backward = [a, b, c = c.get()]()
			{
				a->grad += c->grad;
				b->grad += c->grad;
			};

		return c;
	}

	std::shared_ptr<Scalar> operator*(const std::shared_ptr<Scalar>& a, const std::shared_ptr<Scalar>& b)
	{
		auto out = std::make_shared<Scalar>(a->data * b->data);

		out->node = std::make_shared<Node>("*");
		out->node->parents = { a->node, b->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				a->grad += b->data * out->grad;
				b->grad += a->data * out->grad;
			};

		return out;
	}

	std::shared_ptr<Scalar> negate(const std::shared_ptr<Scalar>& self)
	{
		return self * std::make_shared<Scalar>(-1.0f);
	}

	std::shared_ptr<Scalar> operator-(const std::shared_ptr<Scalar>& a, const std::shared_ptr<Scalar>& b)
	{
		return a + negate(b);
	}

	std::shared_ptr<Scalar> operator^(const std::shared_ptr<Scalar>& a, float b)
	{
		auto c = std::make_shared<Scalar>(std::pow(a->data, b));

		c->node = std::make_shared<Node>("^");
		c->node->parents = { a->node };
		c->node->backward = [a, b, c = c.get()]()
			{
				a->grad += (b * std::pow(a->data, b - 1) * c->grad);
			};

		return c;
	}

	template <typename T>
	concept IsFloats = std::is_same_v<T, float>;

	struct Vector : public std::vector<std::shared_ptr<Scalar>> {
		using std::vector<std::shared_ptr<Scalar>>::vector;

		template<IsFloats... Floats>
		Vector(Floats... args) :
			std::vector<std::shared_ptr<Scalar>>{ std::make_shared<Scalar>(args)... }
		{}

		std::shared_ptr<Node> node{ nullptr };
	};

	std::shared_ptr<Scalar> operator*(const std::vector<std::shared_ptr<Scalar>>& a, const std::vector<std::shared_ptr<Scalar>>& b)
	{
		auto out = std::make_shared<Scalar>(0.0f);
		for (size_t i = 0; i < a.size(); i++) {
			out = out + a[i] * b[i];
		}
		return out;
	}

	std::shared_ptr<Vector> operator+(const std::shared_ptr<Vector>& a, const std::shared_ptr<Vector>& b)
	{
		if (a->size() != b->size()) {
			throw std::runtime_error("a + b: different size.");
		}

		auto out = std::make_shared<Vector>(a->size());
		for (size_t i = 0; i < a->size(); i++) {
			(*out)[i] = std::make_shared<Scalar>((*a)[i]->data + (*b)[i]->data);
		}

		out->node = std::make_shared<Node>("+");
		out->node->parents = { a->node, b->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				for (size_t i = 0; i < a->size(); i++) {
					(*a)[i]->grad += (*out)[i]->grad;
					(*b)[i]->grad += (*out)[i]->grad;
				}
			};

		return out;
	}

	std::shared_ptr<Scalar> operator*(const std::shared_ptr<Vector>& a, const std::shared_ptr<Vector>& b)
	{
		if (a->size() != b->size()) {
			throw std::runtime_error("a * b: different size.");
		}

		auto out = std::make_shared<Scalar>(0.0f);
		for (size_t i = 0; i < a->size(); i++) {
			out->data += (*a)[i]->data * (*b)[i]->data;
		}

		out->node = std::make_shared<Node>("*");
		out->node->parents = { a->node, b->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				for (size_t i = 0; i < a->size(); i++) {
					(*a)[i]->grad += (*b)[i]->data * out->grad;
					(*b)[i]->grad += (*a)[i]->data * out->grad;
				}
			};

		return out;
	}

	std::shared_ptr<Vector> operator*(const std::shared_ptr<Vector>& a, const std::shared_ptr<Scalar>& s)
	{
		auto out = std::make_shared<Vector>(a->size());
		for (size_t i = 0; i < out->size(); i++) {
			(*out)[i]->data = (*a)[i]->data * s->data;
		}

		out->node = std::make_shared<Node>("*");
		out->node->parents = { a->node, s->node };
		out->node->backward = [a, s, out = out.get()]()
			{
				for (size_t i = 0; i < out->size(); i++) {
					(*a)[i]->grad += (*out)[i]->grad * s->data;
					s->grad += (*a)[i]->data * (*out)[i]->grad;
				}
			};

		return out;
	}

	std::shared_ptr<Vector> operator*(const std::shared_ptr<Scalar>& s, const std::shared_ptr<Vector>& a)
	{
		return a * s;
	}

	std::shared_ptr<Vector> negate(const std::shared_ptr<Vector>& self)
	{
		return self * std::make_shared<Scalar>(-1.0f);
	}

	std::shared_ptr<Vector> operator-(const std::shared_ptr<Vector>& a, const std::shared_ptr<Vector>& b)
	{
		return a + negate(b);
	}


	template<typename... Args>
	struct IsSingleFloat {
		static constexpr bool value = false;
	};

	template<>
	struct IsSingleFloat<float> {
		static constexpr bool value = true;
	};

	template<typename... Args>
	struct AreAllFloats {
		static constexpr bool value = (... && std::is_same_v<float, Args>);
	};

	typedef std::variant<std::shared_ptr<Scalar>, std::shared_ptr<Vector>> variant_t;

	struct Tensor {
		template<typename Float, typename = std::enable_if_t<IsSingleFloat<Float>::value>>
		Tensor(Float value)
		{
			this->data.resize(1, value);
			this->grad.resize(1, 0.0f);
		}

		explicit Tensor(size_t size)
		{
			this->data.resize(size, 0.0f);
			this->grad.resize(size, 0.0f);
		}

		template<typename... Floats, typename = std::enable_if_t<AreAllFloats<Floats...>::value && sizeof...(Floats) != 1>>
		Tensor(Floats... args)
		{
			this->data = std::vector<float>(args...);
			this->grad.resize(this->data.size(), 0.0f);
		}

		size_t size() const
		{
			return this->data.size();
		}

		void backward()
		{
			std::vector<std::shared_ptr<Node>> topo;
			std::unordered_set<std::shared_ptr<Node>> visited;

			std::function<void(std::shared_ptr<Node>)> build_topo =
				[&](std::shared_ptr<Node> node)
				{
					if (node && !visited.contains(node)) {
						visited.insert(node);
						for (const auto& parent : node->parents) {
							build_topo(parent);
						}
						topo.emplace_back(node);
					}
				};

			build_topo(this->node);

			std::ranges::fill(this->grad, 1.0f);
			for (auto& v : std::ranges::reverse_view(topo)) {
				v->backward();
			}
		}


		std::vector<float> data;
		std::vector<float> grad;
		std::shared_ptr<Node> node{ nullptr };
	};


	std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& self)
	{
		auto out = std::make_shared<Tensor>(self->size());
		for (size_t i = 0; i < self->size(); i++) {
			out->data[i] = (std::exp(2 * self->data[i]) - 1) / (std::exp(2 * self->data[i]) + 1);
		}

		out->node = std::make_shared<Node>("tanh");
		out->node->parents = { self->node };
		out->node->backward = [self, out = out.get()]()
			{
				for (size_t i = 0; i < self->size(); i++) {
					self->grad[i] += (1.0f - out->data[i] * out->data[i]) * out->grad[i];
				}
			};

		return out;
	}

	std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
	{
		if (a->size() != b->size()) {
			throw std::runtime_error("a + b: different size.");
		}

		auto out = std::make_shared<Tensor>(a->size());
		for (size_t i = 0; i < a->size(); i++) {
			out->data[i] = a->data[i] + b->data[i];
		}

		out->node = std::make_shared<Node>("+");
		out->node->parents = { a->node, b->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				for (size_t i = 0; i < a->size(); i++) {
					a->grad[i] += out->grad[i];
					b->grad[i] += out->grad[i];
				}
			};

		return out;
	}

	std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
	{
		if (a->size() != b->size()) {
			throw std::runtime_error("a * b: different size.");
		}

		auto out = std::make_shared<Tensor>(a->size());
		for (size_t i = 0; i < a->size(); i++) {
			out->data[i] = a->data[i] * b->data[i];
		}

		out->node = std::make_shared<Node>("*");
		out->node->parents = { a->node, b->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				for (size_t i = 0; i < a->size(); i++) {
					a->grad[i] += b->data[i] * out->grad[i];
					b->grad[i] += a->data[i] * out->grad[i];
				}
			};

		return out;
	}

	std::shared_ptr<Tensor> operator^(const std::shared_ptr<Tensor>& a, float b)
	{
		auto out = std::make_shared<Tensor>(a->size());
		for (size_t i = 0; i < a->size(); i++) {
			out->data[i] = std::pow(a->data[i], b);
		}

		out->node = std::make_shared<Node>("^");
		out->node->parents = { a->node };
		out->node->backward = [a, b, out = out.get()]()
			{
				for (size_t i = 0; i < a->size(); i++) {
					a->grad[i] += (b * std::pow(a->data[i], b - 1) * out->grad[i]);
				}
			};

		return out;
	}

	std::shared_ptr<Tensor> negate(const std::shared_ptr<Tensor>& self)
	{
		return self * std::make_shared<Tensor>(-1.0f);
	}

	std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b)
	{
		return a + negate(b);
	}


	class Neuron {
	public:
		Neuron(size_t num_inputs)
		{
			this->bias = std::make_shared<Scalar>(distribution(gen));

			this->weights = std::make_shared<Vector>();
			for (size_t i = 0; i < num_inputs; i++) {
				this->weights->emplace_back(std::make_shared<Scalar>(distribution(gen)));
			}
		}

		std::shared_ptr<Scalar> operator()(const std::shared_ptr<Vector>& x)
		{
			return tanh(this->weights * x + this->bias);
		}

		void parameters(std::function<void(std::shared_ptr<Scalar>&)> f)
		{
			f(this->bias);
			for (auto& weight : *this->weights) {
				f(weight);
			}
		}

		std::shared_ptr<Scalar> bias;
		std::shared_ptr<Vector> weights;
	};

	class Layer {
	public:
		Layer(size_t num_inputs, size_t num_outputs)
		{
			for (size_t i = 0; i < num_outputs; i++) {
				this->neurons.emplace_back(num_inputs);
			}
		}

		std::shared_ptr<Vector> operator()(const std::shared_ptr<Vector>& x)
		{
			auto outs = std::make_shared<Vector>();
			outs->node = std::make_shared<Node>("out");

			for (auto& neuron : this->neurons) {
				outs->emplace_back(neuron(x));
				outs->node->parents.insert(outs->back()->node);
			}
			return outs;
		}

		void parameters(std::function<void(std::shared_ptr<Scalar>&)> f)
		{
			for (auto& neuron : this->neurons) {
				neuron.parameters(f);
			}
		}

		std::vector<Neuron> neurons;
	};

	class MLP {
	public:
		MLP(size_t num_inputs, const std::vector<size_t>& num_outputs)
		{
			std::vector<size_t> sizes = { num_inputs };
			sizes.insert(sizes.end(), num_outputs.begin(), num_outputs.end());

			for (size_t i = 0; i < num_outputs.size(); i++) {
				this->layers.emplace_back(Layer(sizes[i], sizes[i + 1]));
			}
		}

		std::shared_ptr<Vector> operator()(std::shared_ptr<Vector> x)
		{
			for (auto& layer : this->layers) {
				x = layer(x);
			}
			return x;
		}

		void parameters(std::function<void(std::shared_ptr<Scalar>&)> f)
		{
			for (auto& layer : this->layers) {
				layer.parameters(f);
			}
		}

		std::vector<Layer> layers;
	};
}
