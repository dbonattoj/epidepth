#ifndef MF_ASSERT_H_
#define MF_ASSERT_H_

#define MF_STRINGIZE_(X) #X
#define MF_STRINGIZE(X) MF_STRINGIZE_(X)

#define MF_GET_NARG_MACRO_2(_1, _2, NAME, ...) NAME

#ifndef NDEBUG
	#define MF_ASSERT_CRIT_MSG_(__condition__, __msg__) \
		if(! (__condition__)) throw ::mf::failed_assertion(__msg__ " at " __FILE__ ":" MF_STRINGIZE(__LINE__))
	#define MF_ASSERT_MSG_(__condition__, __msg__) \
		if(! (__condition__)) throw ::mf::failed_assertion(__msg__ " at " __FILE__ ":" MF_STRINGIZE(__LINE__))
#else
	#define MF_ASSERT_CRIT_MSG_(__condition__, __msg__) \
		(void)0
	#define MF_ASSERT_MSG_(__condition__, __msg__) \
		if(! (__condition__)) throw ::mf::failed_assertion(__msg__ " at " __FILE__ ":" MF_STRINGIZE(__LINE__))
#endif


#define MF_ASSERT_(__condition__) MF_ASSERT_MSG_(__condition__, "`" #__condition__ "`")
#define MF_ASSERT_CRIT_(__condition__) MF_ASSERT_CRIT_MSG_(__condition__, "`" #__condition__ "`")

#ifdef _MSC_VER
	// workaround for MSVC: http://stackoverflow.com/a/5134656/4108376
	#define MF_EXPAND_(x) x
	#define MF_ASSERT(...) MF_EXPAND_( MF_GET_NARG_MACRO_2(__VA_ARGS__, MF_ASSERT_MSG_, MF_ASSERT_, IGNORE)(__VA_ARGS__) )
	#define MF_ASSERT_CRIT(...) MF_EXPAND_( MF_GET_NARG_MACRO_2(__VA_ARGS__, MF_ASSERT_CRIT_, TFF_ASSERT_CRIT_, IGNORE)(__VA_ARGS__) )
#else
	#define MF_ASSERT(...) MF_GET_NARG_MACRO_2(__VA_ARGS__, MF_ASSERT_MSG_, MF_ASSERT_, IGNORE)(__VA_ARGS__)
	#define MF_ASSERT_CRIT(...) MF_GET_NARG_MACRO_2(__VA_ARGS__, MF_ASSERT_CRIT_MSG_, MF_ASSERT_CRIT_, IGNORE)(__VA_ARGS__)
#endif

#ifdef Assert
	#undef Assert
#endif
#define Assert MF_ASSERT

#ifdef Assert_crit
	#undef Assert_crit
#endif
#define Assert_crit MF_ASSERT_CRIT


///////////////


#include <stdexcept>

namespace mf {

class failed_assertion : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

}

#endif
