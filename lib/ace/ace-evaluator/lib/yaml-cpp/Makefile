SHELL = /bin/sh

# ------ FILES ------

SRC = $(wildcard src/*.cpp)


# ------ DEFINITIONS ------

LIB = libyaml-cpp.a
OBJ =   $(SRC:.cpp=.o)


# ------ SETTINGS ------
CXXFLAGS = -O3 -fPIC -Iinclude

ARCHIVE =	ar
ARCHFLAG =	-rc
USRLIB =
SYSLIB =

# ------ MAKE PROCEDURE ------

lib: $(OBJ)
	$(ARCHIVE) $(ARFLAGS) $(LIB) $(OBJ)

# ------ COMPILE RULES ------

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ------ CLEAN ------
clean: clean-all

clean-all:
	-rm -f *~ $(OBJ) $(LIB)

clean-build:
	-rm -f *~ $(OBJ)
