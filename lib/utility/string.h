/*
Author : Tim Lenertz
Date : May 2016

Copyright (c) 2016, Université libre de Bruxelles

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files to deal in the Software without restriction, including the rights to use, copy, modify, merge,
publish the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MF_UTILITY_STRING_H_
#define MF_UTILITY_STRING_H_

#include <string>
#include <vector>

namespace mf {

std::string file_name_extension(const std::string& filename);
std::string replace_file_name_extension(const std::string& filename, const std::string& ext);
std::string remove_file_name_extension(const std::string& filename);

template<typename T> std::string to_string(const T&);
template<typename T> T from_string(const std::string&);

template<typename It> std::string to_string(It begin, It end, const std::string& separator = ", ");

std::vector<std::string> explode(char separator, const std::string&);
std::string implode(char separator, const std::vector<std::string>&);

template<typename T>
std::vector<T> explode_from_string(char separator, const std::string&);

template<typename T>
std::string implode_to_string(char separator, const std::vector<T>&);

std::string to_lower(const std::string&);
std::string to_upper(const std::string&);

std::string replace_all(const std::string& subject, const std::string& find, const std::string& replace);
std::size_t replace_all_inplace(std::string& subject, const std::string& find, const std::string& replace);

}

#include "string.tcc"

#endif
