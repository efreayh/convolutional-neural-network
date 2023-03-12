#include <string>
#include <cctype>
#include <algorithm>
#include "utility.hpp"

bool utility::compare_ignore_case(std::string s1, std::string s2) {
    std::transform(s1.begin(), s1.end(), s1.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(s2.begin(), s2.end(), s2.begin(), [](unsigned char c) { return std::tolower(c); });

    return s1 == s2;
}