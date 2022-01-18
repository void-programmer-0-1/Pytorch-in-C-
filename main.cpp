
#include <stdlib.h>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h> 

struct NeuralNet : torch::nn::Module{

    torch::nn::Linear l1{nullptr};
    torch::nn::Linear l2{nullptr};

    NeuralNet()
        :l1(1,1),l2(1,1){
            register_module("l1",l1);
            register_module("l2",l2);
        }

    torch::Tensor forward(torch::Tensor x){
        x = l1(x);
        x = l2(x);
        return x;
    }

};

int main(){

    torch::jit::script::Module module;
    module = torch::jit::load("../linear_regression.pt");

    std::vector<torch::jit::IValue> inputs;

    float user_data;
    std::cout << "Enter a data :: ";
    std::cin >> user_data;

    torch::Tensor data = torch::tensor({user_data}, torch::kFloat);
    inputs.push_back(data);
    
    torch::Tensor prediction = module.forward(inputs).toTensor();

    std::cout << prediction << std::endl;

}