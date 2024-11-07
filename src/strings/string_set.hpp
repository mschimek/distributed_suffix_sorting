#pragma once

#include <string>
#include <utility>

#include "tlx/sort/strings/string_set.hpp"


namespace dsss {

/*!
 * Class implementing StringSet concept for any struct that wraps a string.
 * The struct has to define cbegin_chars, cend_chars, get_string.
 */
template <typename StringType>
class GeneralStringSetTraits {
public:
    //! exported alias for character type
    typedef StringType::CharType Char;

    //! String reference: StringType, which should be reference counted.
    typedef StringType String;

    //! Iterator over string references: pointer to std::string.
    typedef StringType* Iterator;

    //! iterator of characters in a string
    typedef const Char* CharIterator;

    //! exported alias for assumed string container
    typedef std::pair<Iterator, size_t> Container;
};

/*!
 * Class implementing StringSet concept for structs.
 */
template <typename StringType>
class GeneralStringSet
    : public GeneralStringSetTraits<StringType>,
      public tlx::sort_strings_detail::StringSetBase<GeneralStringSet<StringType>,
                                                     GeneralStringSetTraits<StringType>> {
public:
    typedef GeneralStringSet<StringType> GeneralStringSet_;
    typedef GeneralStringSetTraits<StringType> Parent;
    typedef Parent::Char Char;
    typedef Parent::String String;
    typedef Parent::Iterator Iterator;
    typedef Parent::CharIterator CharIterator;
    typedef Parent::Container Container;

    //! Construct from begin and end string pointers
    GeneralStringSet(const Iterator& begin, const Iterator& end) : begin_(begin), end_(end) {}

    //! Construct from a string container
    explicit GeneralStringSet(Container& c) : begin_(c.first), end_(c.first + c.second) {}

    //! Return size of string array
    size_t size() const { return end_ - begin_; }

    //! Iterator representing first String position
    Iterator begin() const { return begin_; }

    //! Iterator representing beyond last String position
    Iterator end() const { return end_; }

    //! Array access (readable and writable) to String objects.
    String& operator[](const Iterator& i) const { return *i; }

    //! Return CharIterator for referenced string, which belongs to this set.
    static CharIterator get_chars(const String& s, size_t depth) {
        return s.cbegin_chars() + depth;
    }

    //! Returns true if CharIterator is at end of the given String
    static bool is_end(const String& s, const CharIterator& i) { return i == s.cend_chars(); }

    //! Return complete string (for debugging purposes)
    static std::string get_string(const String& s, size_t depth = 0) { return s.get_string(); }

    //! Subset this string set using iterator range.
    static GeneralStringSet_ sub(Iterator begin, Iterator end) {
        return GeneralStringSet_(begin, end);
    }

    //! Allocate a new temporary string container with n empty Strings
    static Container allocate(size_t n) { return std::make_pair(new String[n], n); }

    //! Deallocate a temporary string container
    static void deallocate(Container& c) {
        delete[] c.first;
        c.first = nullptr;
    }

private:
    //! pointers to std::string objects
    Iterator begin_, end_;
};

} // namespace dsss