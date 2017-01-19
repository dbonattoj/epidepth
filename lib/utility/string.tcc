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

#include <sstream>
#include <typeinfo>

namespace mf {

template<typename T>
std::string to_string(const T& t) {
	std::ostringstream str;
	str << t;
	return str.str();
}


template<typename It> std::string to_string(It begin, It end, const std::string& separator) {
	std::ostringstream str;
	for(It it = begin; it != end; ++it) {
		if(it != begin) str << separator;
		str << *it;
	}
	return str.str();
}


template<typename T>
T from_string(const std::string& s) {
	std::istringstream str(s);
	T t;
	str >> t;
	if(str.fail()) throw std::invalid_argument("cannot convert \"" + s + "\" to type " + typeid(T).name() + ".");
	else return t;
}


template<typename T>
std::vector<T> explode_from_string(char separator, const std::string& str) {
	std::vector<std::string> vec = explode(separator, str);
	std::vector<T> vec2(vec.size());
	auto it2 = vec2.begin();
	for(const std::string& s : vec) *(it2++) = from_string<T>(s);
	return vec2;
}


template<typename T>
std::string implode_to_string(char separator, const std::vector<T>& vec) {
	std::vector<std::string> vec2(vec.size());
	auto it2 = vec2.begin();
	for(const T& t : vec) *(it2++) = to_string(t);
	return implode(separator, vec2);
}


}
