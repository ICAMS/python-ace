//
// Created by Lysogorskiy Yury on 27.03.2020.
//
#include "util_generate_coupling.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_coupling.yaml> [<output.yaml>]\n", argv[0]);
        exit(EXIT_SUCCESS);
    }
    string inputfile = argv[1];


    if (!if_file_exist(inputfile)) {
        cout << "Potential file " << inputfile << " doesn't exists" << endl;
        exit(EXIT_FAILURE);
    }
    string outputfile = "";
    if (argc > 2) {
        outputfile = argv[2];
    }

    process_coupling_constans(inputfile, outputfile);

    return 0;
}
