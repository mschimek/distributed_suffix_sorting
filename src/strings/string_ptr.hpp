#pragma once

#include <cstdint>
#include "strings/lcp_type.hpp"

// adapted from:
// https://github.com/pmehnert/distributed-string-sorting/blob/master/src/merge/stringptr.hpp

namespace dsss {

template <typename _StringSet>
class StringLcpPtrMergeAdapter {
public:
    typedef _StringSet StringSet;
    typedef typename StringSet::String String;
    using Iterator = typename StringSet::Iterator;
    using CharIt = typename StringSet::CharIterator;

public:
    StringSet active_;
    LcpType* lcp_;
    uint64_t offset;
    Iterator strings_;

public:
    StringLcpPtrMergeAdapter()
        : active_(StringSet(nullptr, nullptr)),
          lcp_(nullptr),
          offset(0),
          strings_(nullptr) {}
    //! constructor specifying all attributes
    StringLcpPtrMergeAdapter(StringSet const& ss, LcpType* lcp_begin)
        : active_(ss),
          lcp_(lcp_begin),
          offset(0),
          strings_(ss.begin()) {}

    //! return currently active array
    StringSet const& active() const { return active_; }

    ////! return valid length
    uint64_t size() const { return active_.size() - offset; }

    ////! Advance (both) pointers by given offset, return sub-array
    StringLcpPtrMergeAdapter sub(uint64_t _offset, uint64_t _size) const {
        return StringLcpPtrMergeAdapter(active_.subi(offset + _offset, offset + _offset + _size),
                                        lcp_ + _offset + offset);
    }

    bool empty() const { return (active().size() - offset <= 0); }

    void setFirst(String str, LcpType lcp) {
        *(strings_ + offset) = str;
        *(lcp_ + offset) = lcp;
    }

    String& firstString() const { return *(strings_ + offset); }

    CharIt firstStringChars() const { return active().get_chars(*(strings_ + offset), 0); }

    LcpType& firstLcp() const { return *(lcp_ + offset); }

    StringLcpPtrMergeAdapter& operator++() {
        ++offset;
        return *this;
    }

    bool operator<(StringLcpPtrMergeAdapter const& rhs) const {
        return strings_ + offset < rhs.strings_ + rhs.offset;
    }
};
} // namespace dsss