# OpenMP not supported by gcc on mac, instead gcc has to be fixed using the below code. Once compiled this shouldnt be a problem. 
#
# So for users of the software this will not effect the usability of the software. To fix it to have OpenMP working on Mac just run the below
#brew code;
brew reinstall gcc --without-multilib
