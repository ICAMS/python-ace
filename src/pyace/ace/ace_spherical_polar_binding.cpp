#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <string>
#include "ace_spherical_polar.h"

namespace py = pybind11;
using namespace std;

/*
This is the Py Class definition which inherits from the main class

The inherited class will contain extra functions
*/
class PyACESHarmonics: public ACESHarmonics {
 public:
   //constructor for the basis function
   int lmax;
   explicit PyACESHarmonics(int lm) : ACESHarmonics(lm) { lmax=lm; }
   //access functions for plm and ylm
   vector<vector<double>> get_plm();
   void set_plm();
   //vector<vector<double>> get_ylm();
   //vector<vector<double>> set_ylm();

};

vector<vector<double>> PyACESHarmonics::get_plm(){
//  auto res =  //convert_array2dlm_to_vec_double(plm, lmax);
  return plm.to_vector();
}

void PyACESHarmonics::set_plm(){
    //empty for now
}

//finally add bindings
PYBIND11_MODULE(sharmonics, m) {
    py::options options;
    options.disable_function_signatures();

//Bindings for the ace class
//------------------------------------------------------------------
py::class_<ACESHarmonics>(m,"ACESHarmonics", R"mydelimiter(

    )mydelimiter")

    .def(py::init < int >(), py::arg("lmax")= 10)
    .def("compute_plm",&ACESHarmonics::compute_plm)
    ;

//this is the derived class bindings
py::class_<PyACESHarmonics, ACESHarmonics>(m,"PyACESHarmonics", R"mydelimiter(

    )mydelimiter")

    .def(py::init < int >(), py::arg("lmax")= 10)
    .def_property("plm",&PyACESHarmonics::get_plm, &PyACESHarmonics::set_plm )
    ;


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
