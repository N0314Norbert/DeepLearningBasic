


#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <any>
#include <typeinfo>
#include <type_traits>
#include <unordered_map>
#include <random>

using namespace std;

class Matrix
{
public:
	int rows;
	int col;

	vector<vector<float>> matrix;
	
	
	Matrix(int _rows, int _col)
	{
		this->rows = _rows;
		this->col = _col;
		vector<float> sub;
		for (int i = 0; i < _rows; i++)
		{
			for (int j = 0; j < _col; j++)
			{
				sub.push_back(0);
			}
			this->matrix.push_back(sub);
			sub.clear();
		}
	}

	void randomize(vector<vector<float>> &matrix)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->col; j++)
			{
				srand(time(NULL));

				matrix[i][j] = rand() % 3 - 1;
			}
		}
	}
	vector<float> toArray(vector<vector<float>> matrix)
	{
		vector<float> resault;

		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->col; j++)
			{
				resault.push_back(matrix[i][j]);
			}
		}
		return resault;

	}

	

	void add(vector<vector<float>> &value, vector<vector<float>>other)
	{
		for (int i = 0; i < value.size(); i++)
		{
			for (int j = 0; j < value[i].size(); j++)
			{
				value[i][j] += value[i][j];
			}
		}
	}
	void add(float value)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->col; j++)
			{
				this->matrix[i][j] += value;
			}
		}
	}
	vector<vector<float>> transpose(vector<vector<float>> matrix)
	{
		Matrix nn(matrix[0].size(),matrix.size());
		nn.randomize(nn.matrix);
		vector<vector<float>> resault = nn.matrix;
		
		for (int i = 0; i < resault[0].size(); i++)
		{
			for (int j = 0; j < resault.size(); j++)
			{
				resault[j][i] = matrix[i][j];
			}
		}
		return resault;
	}
	vector<vector<float>> multiply(vector<vector<float>> otherA, vector<vector<float>> otherB)
	{
		vector<vector<float>> resault;
		vector<float> sub;
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->col; j++)
			{
				sub.push_back(otherA[i][j] *= otherB[i][j]);
			}
			resault.push_back(sub);
			sub.clear();
		}
		return resault;
	}
	void multiply(float &other)
	{
		for (int i = 0; i < this->rows; i++)
		{
			for (int j = 0; j < this->col; j++)
			{
				this->matrix[i][j] *= other;
			}
		}
	}
	vector<vector<float>> subtract(vector<vector<float>> a, vector<vector<float>> b)
	{
		vector<vector<float>> resault;
		vector<float> sub;
		for (int i = 0; i < a.size(); i++)
		{
			for (int j = 0; j < a[i].size(); j++)
			{
				sub.push_back(a[i][j] - b[i][j]);
			}
			resault.push_back(sub);
			sub.clear();
		}
		return resault;
	}
	vector<vector<float>> dot(vector<vector<float>> a, vector<vector<float>> b)
	{
		if (a[0].size() != b.size())
		{
			std::cout << "error" << endl;
			return {};
		}

		vector<vector<float>> resault;
		vector<float> sub;

		for (int i = 0; i < a.size(); i++)
		{
			for (int j = 0; j < b[0].size(); j++)
			{
				float sum = 0;
				for (int k = 0; k < a[0].size(); k++)
				{
					sum += a[i][k] * b[k][j];
				}
				sub.push_back(sum);
			}
			resault.push_back(sub);
			sub.clear();
		}
		return resault;
	}
};

class NeuralNetwork
{
public:

	int inputs;
	int hidden;
	int outputs;

	float learningRate = 0.001f;

	vector<vector<float>> weights_ih;
	vector<vector<float>> weights_ho;

	vector<vector<float>> bias_h;
	vector<vector<float>> bias_o;

	

	NeuralNetwork(int _inputs, int _hidden, int _outputs)
	{
		this->inputs = _inputs;
		this->hidden = _hidden;
		this->outputs = _outputs;

		
	}
	float dsigmoid(float y)
	{
		return y = y * (1 - y);
	}

	vector<float> feedforward(vector<vector<float>> input)
	{
		Matrix matrix(1,1);
		vector<vector<float>> hidden = matrix.dot(weights_ih, input);
		matrix.add(hidden, bias_h);

		vector<vector<float>> sigmoid_h;
		vector<float> sub;

		for (int i = 0; i < hidden.size(); i++)
		{
			for (int j = 0; j < hidden[i].size(); j++)
			{

				sub.push_back( 1 / (1 + exp(-1 * hidden[i][j])));
			}
			sigmoid_h.push_back(sub);
			sub.clear();
		}

		vector<vector<float>> output = matrix.dot(weights_ho, sigmoid_h);
		matrix.add(output, bias_o);

		vector<vector<float>> sigmoid_o;

		for (int i = 0; i < output.size(); i++)
		{
			for (int j = 0; j < output[i].size(); j++)
			{

				sub.push_back(1 / (1 + exp(-1 * output[i][j])));
			}
			sigmoid_o.push_back(sub);
			sub.clear();
		}

		vector<float> resault = matrix.toArray(sigmoid_o);
		
		return resault;
	}

	void train(vector<vector<float>> inputs, vector<vector<float>> targets)
	{
		Matrix matrix(1, 1);

		vector<vector<float>> outputs;
		vector<vector<float>> input = inputs;
		vector<float> sub = feedforward(input);
		outputs.push_back(sub);

		vector<vector<float>> output_error = matrix.subtract(targets, outputs);

		vector<vector<float>> who_t = matrix.transpose(weights_ho);
		vector<vector<float>> hidden = matrix.dot(who_t, output_error);

		for (size_t i = 0; i < outputs.size(); i++)
		{
			for (size_t j = 0; j < outputs[i].size(); j++)
			{
				outputs[i][j] = dsigmoid(outputs[i][j]);
			}
		}

		vector<vector<float>>outputsM = matrix.multiply(outputs, output_error);
		for (size_t i = 0; i < outputsM.size(); i++)
		{
			for (size_t j = 0; j < outputsM[i].size(); j++)
			{
				outputsM[i][j] = outputsM[i][j] * learningRate;
			}
		}
		matrix.add(bias_o, outputsM);

		vector<vector<float>> hidden_T = matrix.transpose(hidden);

		vector<vector<float>> weight_ho_deltas = matrix.dot(outputsM, hidden_T);

		for (size_t i = 0; i < weight_ho_deltas.size(); i++)
		{
			for (size_t j = 0; j < weight_ho_deltas[i].size(); j++)
			{
				weights_ho[i][j] = weight_ho_deltas[i][j] + weights_ho[i][j];
			}
		}

		vector<vector<float>> hidden_gradient = hidden;

		for (size_t i = 0; i < hidden.size(); i++)
		{
			for (size_t j = 0; j < hidden[i].size(); j++)
			{
				hidden_gradient[i][j] = dsigmoid(hidden[i][j]);
			}
		}
		vector<vector<float>> hidden_errors = matrix.dot(who_t, output_error);
		
		hidden_gradient = matrix.multiply(hidden_errors, hidden_gradient);

		for (size_t i = 0; i < hidden_gradient.size(); i++)
		{
			for (size_t j = 0; j < hidden_gradient[i].size(); j++)
			{
				hidden_gradient[i][j] = hidden_gradient[i][j] * learningRate;
			}
		}
		matrix.add(bias_h, hidden_gradient);

		vector<vector<float>> inputs_T = matrix.transpose(inputs);

		vector<vector<float>> weight_ih_deltas = matrix.dot(hidden_gradient, inputs_T);

		for (size_t i = 0; i < weight_ih_deltas.size(); i++)
		{
			for (size_t j = 0; j < weight_ih_deltas[i].size(); j++)
			{

				weights_ih[i][j] = weight_ih_deltas[i][j] + weights_ih[i][j];
			}
		}
		
	}

};

int main()
{
	NeuralNetwork nn(2,2,1);
	Matrix matrix(nn.hidden,nn.inputs);
	Matrix output(nn.outputs, nn.hidden);

	matrix.randomize(matrix.matrix);
	output.randomize(output.matrix);

	nn.weights_ih = matrix.matrix;
	nn.weights_ho = output.matrix;

	nn.bias_h = { {1, 0.4 } };
	nn.bias_o = { {0.25, 0.4} };


	vector<vector<float>> input = { {0}, {1} };
	vector<vector<float>> targets = { {1} };

	mt19937 generator;
	generator.seed(time(0));

	std::uniform_int_distribution<uint32_t> dice(0,3);

	for (size_t i = 0; i < 60000; i++)
	{

		switch (dice(generator))
		{

		case 0:
			nn.train(input, targets);
			break;
		case 1:
			nn.train({ {1}, {0} }, targets);
			break;
		case 2:
			nn.train({ {1}, {1} }, { {0} });
			break;
		case 3:
			nn.train({ {0}, {0} }, { {0} });
			break;
		default:
			std::cout << "default" << endl;
			break;
		}
	}

	std::cout << nn.feedforward(input)[0] << endl;
	std::cout << nn.feedforward({ {1}, {0} })[0] << endl;
	std::cout << nn.feedforward({ {1}, {1} })[0] << endl;
	std::cout << nn.feedforward({ {0}, {0} })[0] << endl;





	return 0;
}

