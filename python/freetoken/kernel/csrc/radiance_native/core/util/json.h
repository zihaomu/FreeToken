/* util/json.h -- the ordered JSON DOM the server, chat templates and config files share.
 *
 * It exists because the reader matters: request bodies are parsed with yyjson
 * (vendor/yyjson) at several GB/s, where the DOM library this replaced read them at
 * ~340 MB/s, and a DOM vendored around another library's parser would still have owed that
 * parser its value model. The model kept is that predecessor's, where the tree's code can
 * observe it -- insertion-ordered objects, non-negative integers read as unsigned, an
 * integer past uint64 read as a double, duplicate keys keeping the first position with the
 * last value, dump() bytes for a parsed value -- and the type is otherwise the smallest
 * value type that carries the API the tree uses. tests/json_test.cpp pins the model on the
 * corpora that distinguish readers; json.cpp says where behaviour still differs from the
 * predecessor's (error texts, and a float's digits where the predecessor's were not the
 * shortest that read back).
 *
 * Numbers are int64, uint64 or double, never coerced at rest. get<T>() converts between
 * number kinds and throws Json::exception across kind boundaries (a string is not a
 * number). Iteration over an object yields its values in key order with it.key() naming
 * each; iteration over a scalar yields the scalar itself.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <iosfwd>
#include <iterator>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace rad {

namespace json_detail {
/* The container types the implicit conversion operator may produce. Deliberately a closed
 * list: a structural "has value_type" test also claims std::optional and
 * std::initializer_list<char>, whose conversions then tie with ordinary assignments. */
template <class T>
struct is_gettable_container : std::false_type {};
template <class U, class A>
struct is_gettable_container<std::vector<U, A>> : std::true_type {};
template <class U, class A>
struct is_gettable_container<std::deque<U, A>> : std::true_type {};
template <class U, class C, class A>
struct is_gettable_container<std::set<U, C, A>> : std::true_type {};
template <class U, class H, class E, class A>
struct is_gettable_container<std::unordered_set<U, H, E, A>> : std::true_type {};
template <class U, class C, class A>
struct is_gettable_container<std::map<std::string, U, C, A>> : std::true_type {};
template <class U, class H, class E, class A>
struct is_gettable_container<std::unordered_map<std::string, U, H, E, A>> : std::true_type {};
}  /* namespace json_detail */

class Json {
public:
    enum class value_t {
        null,
        boolean,
        number_integer,
        number_unsigned,
        number_float,
        object,
        array,
        string,
        discarded,
    };

    /* One exception type, thrown for the type errors, missing keys and parse failures
     * the predecessor reported with its own hierarchy. Callers caught std::exception or
     * json::exception; both still work. The texts are this library's own. */
    struct exception : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    enum class error_handler_t { strict, replace, ignore };

    using array_t = std::vector<Json>;
    using object_t = std::vector<std::pair<std::string, Json>>;
    using string_t = std::string;

private:
    struct discarded_t {};

    using storage_t = std::variant<std::monostate /* null */, bool, int64_t, uint64_t, double,
                                   string_t, array_t, object_t, discarded_t>;
    storage_t v_;

    [[noreturn]] static void fail(const char* what) { throw exception(what); }
    [[noreturn]] void fail_type(const char* want) const;

    bool is_number_kind() const {
        return std::holds_alternative<int64_t>(v_) || std::holds_alternative<uint64_t>(v_) ||
               std::holds_alternative<double>(v_);
    }
    long double as_long_double() const {
        if (const auto* p = std::get_if<int64_t>(&v_)) return (long double)*p;
        if (const auto* p = std::get_if<uint64_t>(&v_)) return (long double)*p;
        return (long double)std::get<double>(v_);
    }

public:
    Json() = default;
    /* Assignment copies (or moves) the source out before the old value dies: the source is
     * routinely a child of the target (`n = n["$defs"]`), which the variant's own assignment
     * would read after destroying. the predecessor's assignment was safe the same way. */
    Json(const Json&) = default;
    Json(Json&&) = default;
    Json& operator=(const Json& o) {
        if (this != &o) {
            storage_t tmp = o.v_;
            v_ = std::move(tmp);
        }
        return *this;
    }
    Json& operator=(Json&& o) {
        if (this != &o) {
            storage_t tmp = std::move(o.v_);
            v_ = std::move(tmp);
        }
        return *this;
    }
    Json(std::nullptr_t) {}
    Json(value_t t) {
        switch (t) {
        case value_t::boolean: v_ = false; break;
        case value_t::number_integer: v_ = int64_t(0); break;
        case value_t::number_unsigned: v_ = uint64_t(0); break;
        case value_t::number_float: v_ = 0.0; break;
        case value_t::string: v_ = string_t(); break;
        case value_t::array: v_ = array_t(); break;
        case value_t::object: v_ = object_t(); break;
        case value_t::discarded: v_ = discarded_t(); break;
        case value_t::null: break;
        }
    }
    Json(bool b) : v_(b) {}
    Json(int i) : v_((int64_t)i) {}
    Json(int64_t i) : v_(i) {}
    Json(long long i) : v_((int64_t)i) {}
    Json(unsigned u) : v_((uint64_t)u) {}
    Json(uint64_t u) : v_(u) {}
    Json(unsigned long long u) : v_((uint64_t)u) {}
    Json(double d) : v_(d) {}
    Json(float f) : v_((double)f) {}
    Json(const char* s) : v_(string_t(s)) {}
    Json(string_t s) : v_(std::move(s)) {}
    Json(std::string_view s) : v_(string_t(s)) {}
    Json(array_t a) : v_(std::move(a)) {}
    Json(object_t o) : v_(std::move(o)) {}

    /* Container construction mirrors to_json on the obvious containers: a vector becomes an
     * array of converted elements, a string-keyed map an object. */
    template <class T, std::enable_if_t<!std::is_same_v<std::decay_t<T>, Json> &&
                                        !std::is_same_v<std::decay_t<T>, array_t> &&
                                        !std::is_same_v<std::decay_t<T>, std::vector<uint8_t>>, int> = 0>
    Json(const std::vector<T>& a) {
        array_t out;
        out.reserve(a.size());
        for (const T& e : a) out.emplace_back(e);
        v_ = std::move(out);
    }
    template <class M, std::enable_if_t<requires { typename M::mapped_type; } &&
                                        std::is_same_v<typename M::key_type, std::string>, int> = 0>
    Json(const M& m) {
        object_t out;
        out.reserve(m.size());
        for (const auto& kv : m) out.emplace_back(kv.first, Json(kv.second));
        v_ = std::move(out);
    }

    /* the predecessor's rule: a list whose every element is a two-element array with a string first
     * is an object; anything else is an array. An empty list is an array. */
    Json(std::initializer_list<Json> init) {
        bool obj = init.size() > 0;
        for (const Json& e : init) {
            const auto* a = std::get_if<array_t>(&e.v_);
            if (!a || a->size() != 2 || !std::holds_alternative<string_t>((*a)[0].v_)) {
                obj = false;
                break;
            }
        }
        if (obj) {
            object_t out;
            out.reserve(init.size());
            for (const Json& e : init) {
                const array_t& a = std::get<array_t>(e.v_);
                out.emplace_back(std::get<string_t>(a[0].v_), a[1]);
            }
            v_ = std::move(out);
        } else {
            v_ = array_t(init.begin(), init.end());
        }
    }

    static Json array(std::initializer_list<Json> init = {}) {
        Json j;
        j.v_ = array_t(init.begin(), init.end());
        return j;
    }
    static Json object(std::initializer_list<Json> init = {}) {
        Json j;
        j.v_ = object_t();
        for (const Json& e : init) {
            const auto* a = std::get_if<array_t>(&e.v_);
            if (!a || a->size() != 2 || !std::holds_alternative<string_t>((*a)[0].v_))
                fail("json::object() needs two-element {key, value} entries");
            std::get<object_t>(j.v_).emplace_back(std::get<string_t>((*a)[0].v_), (*a)[1]);
        }
        return j;
    }

    /* ------------------------------------------------------------------ kinds */
    value_t type() const {
        switch (v_.index()) {
        case 0: return value_t::null;
        case 1: return value_t::boolean;
        case 2: return value_t::number_integer;
        case 3: return value_t::number_unsigned;
        case 4: return value_t::number_float;
        case 5: return value_t::string;
        case 6: return value_t::array;
        case 7: return value_t::object;
        default: return value_t::discarded;
        }
    }
    const char* type_name() const {
        switch (type()) {
        case value_t::null: return "null";
        case value_t::boolean: return "boolean";
        case value_t::string: return "string";
        case value_t::array: return "array";
        case value_t::object: return "object";
        case value_t::discarded: return "discarded";
        default: return "number";
        }
    }
    bool is_null() const { return std::holds_alternative<std::monostate>(v_); }
    bool is_boolean() const { return std::holds_alternative<bool>(v_); }
    bool is_number() const { return is_number_kind(); }
    bool is_number_integer() const {
        return std::holds_alternative<int64_t>(v_) || std::holds_alternative<uint64_t>(v_);
    }
    bool is_number_unsigned() const { return std::holds_alternative<uint64_t>(v_); }
    bool is_number_float() const { return std::holds_alternative<double>(v_); }
    bool is_string() const { return std::holds_alternative<string_t>(v_); }
    bool is_array() const { return std::holds_alternative<array_t>(v_); }
    bool is_object() const { return std::holds_alternative<object_t>(v_); }
    bool is_discarded() const { return std::holds_alternative<discarded_t>(v_); }
    bool is_primitive() const { return is_null() || is_boolean() || is_number() || is_string(); }
    bool is_structured() const { return is_array() || is_object(); }

    size_t size() const {
        if (const auto* a = std::get_if<array_t>(&v_)) return a->size();
        if (const auto* o = std::get_if<object_t>(&v_)) return o->size();
        return is_null() ? 0 : 1;
    }
    bool empty() const { return size() == 0; }
    void clear() {
        if (auto* a = std::get_if<array_t>(&v_)) a->clear();
        else if (auto* o = std::get_if<object_t>(&v_)) o->clear();
        else if (auto* s = std::get_if<string_t>(&v_)) s->clear();
    }

    /* ------------------------------------------------------------------ iterators
     * One iterator shape over arrays, objects and (as the predecessor's did) a lone scalar. key()
     * names the member for objects and the index as a string for arrays, as the predecessor's did. */
    template <bool Const>
    class iter_impl {
        friend class Json;
        template <bool B>
        friend class iter_impl;
        using J = std::conditional_t<Const, const Json, Json>;
        using arr_it = std::conditional_t<Const, array_t::const_iterator, array_t::iterator>;
        using obj_it = std::conditional_t<Const, object_t::const_iterator, object_t::iterator>;

        J* owner_ = nullptr;
        arr_it a_{};
        obj_it o_{};
        size_t scalar_ = 0;  /* 0: at the scalar itself, 1: past it */

        static iter_impl for_array(J* owner, arr_it it) {
            iter_impl r;
            r.owner_ = owner;
            r.a_ = it;
            return r;
        }
        static iter_impl for_object(J* owner, obj_it it) {
            iter_impl r;
            r.owner_ = owner;
            r.o_ = it;
            return r;
        }
        static iter_impl for_scalar(J* owner, size_t at) {
            iter_impl r;
            r.owner_ = owner;
            r.scalar_ = at;
            return r;
        }

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Json;
        using difference_type = std::ptrdiff_t;
        using pointer = J*;
        using reference = J&;

        iter_impl() = default;
        /* iterator converts to const_iterator */
        template <bool C2, std::enable_if_t<Const && !C2, int> = 0>
        iter_impl(const iter_impl<C2>& o) : owner_(o.owner_), a_(o.a_), o_(o.o_), scalar_(o.scalar_) {}

        reference operator*() const {
            if (owner_->is_array()) return *a_;
            if (owner_->is_object()) return o_->second;
            return *owner_;
        }
        pointer operator->() const { return &operator*(); }

        std::string key() const {
            if (owner_->is_object()) return o_->first;
            if (owner_->is_array())
                return std::to_string((size_t)(a_ - const_cast<J*>(owner_)->arr().begin()));
            return std::string();
        }
        reference value() const { return operator*(); }

        iter_impl& operator++() {
            if (owner_->is_array()) ++a_;
            else if (owner_->is_object()) ++o_;
            else ++scalar_;
            return *this;
        }
        iter_impl operator++(int) {
            iter_impl t = *this;
            ++*this;
            return t;
        }
        iter_impl& operator--() {
            if (owner_->is_array()) --a_;
            else if (owner_->is_object()) --o_;
            else --scalar_;
            return *this;
        }
        iter_impl operator--(int) {
            iter_impl t = *this;
            --*this;
            return t;
        }
        iter_impl& operator+=(difference_type n) {
            if (owner_->is_array()) a_ += n;
            else if (owner_->is_object()) o_ += n;
            else scalar_ = (size_t)((difference_type)scalar_ + n);
            return *this;
        }
        iter_impl& operator-=(difference_type n) { return *this += -n; }
        iter_impl operator+(difference_type n) const {
            iter_impl t = *this;
            return t += n;
        }
        iter_impl operator-(difference_type n) const {
            iter_impl t = *this;
            return t -= n;
        }
        difference_type operator-(const iter_impl& o) const {
            if (owner_->is_array()) return a_ - o.a_;
            if (owner_->is_object()) return o_ - o.o_;
            return (difference_type)scalar_ - (difference_type)o.scalar_;
        }
        bool operator==(const iter_impl& o) const {
            if (owner_ != o.owner_) return false;
            if (!owner_) return true;
            if (owner_->is_array()) return a_ == o.a_;
            if (owner_->is_object()) return o_ == o.o_;
            return scalar_ == o.scalar_;
        }
        bool operator!=(const iter_impl& o) const { return !(*this == o); }
        bool operator<(const iter_impl& o) const { return (*this - o) < 0; }
    };

    using iterator = iter_impl<false>;
    using const_iterator = iter_impl<true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
    array_t& arr() { return std::get<array_t>(v_); }
    const array_t& arr() const { return std::get<array_t>(v_); }
    object_t& obj() { return std::get<object_t>(v_); }
    const object_t& obj() const { return std::get<object_t>(v_); }

public:
    iterator begin() {
        if (is_array()) return iterator::for_array(this, arr().begin());
        if (is_object()) return iterator::for_object(this, obj().begin());
        return iterator::for_scalar(this, is_null() ? 1 : 0);
    }
    iterator end() {
        if (is_array()) return iterator::for_array(this, arr().end());
        if (is_object()) return iterator::for_object(this, obj().end());
        return iterator::for_scalar(this, 1);
    }
    const_iterator begin() const {
        if (is_array()) return const_iterator::for_array(this, arr().begin());
        if (is_object()) return const_iterator::for_object(this, obj().begin());
        return const_iterator::for_scalar(this, is_null() ? 1 : 0);
    }
    const_iterator end() const {
        if (is_array()) return const_iterator::for_array(this, arr().end());
        if (is_object()) return const_iterator::for_object(this, obj().end());
        return const_iterator::for_scalar(this, 1);
    }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const { return rbegin(); }
    const_reverse_iterator crend() const { return rend(); }

    /* items(): object members as key/value proxies, array elements under index keys, as
     * the predecessor's iteration proxy presented them. */
    template <bool Const>
    struct items_proxy {
        using base_it = iter_impl<Const>;
        struct proxy {
            base_it it;
            std::string key() const { return it.key(); }
            typename base_it::reference value() const { return it.value(); }
            /* Structured bindings: for (auto& [k, v] : obj.items()). */
            template <size_t I>
            decltype(auto) get() const {
                if constexpr (I == 0) return key();
                else return value();
            }
        };
        struct iterator {
            base_it it;
            proxy operator*() const { return proxy{ it }; }
            iterator& operator++() {
                ++it;
                return *this;
            }
            iterator operator++(int) {
                iterator t = *this;
                ++it;
                return t;
            }
            bool operator==(const iterator& o) const { return it == o.it; }
            bool operator!=(const iterator& o) const { return it != o.it; }
        };
        base_it b, e;
        iterator begin() const { return iterator{ b }; }
        iterator end() const { return iterator{ e }; }
    };
    items_proxy<false> items() { return { begin(), end() }; }
    items_proxy<true> items() const { return { begin(), end() }; }

    /* ------------------------------------------------------------------ access */
    iterator find(const std::string& key) {
        if (!is_object()) return end();
        auto& o = obj();
        for (auto it = o.begin(); it != o.end(); ++it)
            if (it->first == key) return iterator::for_object(this, it);
        return end();
    }
    const_iterator find(const std::string& key) const {
        if (!is_object()) return end();
        const auto& o = obj();
        for (auto it = o.begin(); it != o.end(); ++it)
            if (it->first == key) return const_iterator::for_object(this, it);
        return end();
    }
    iterator find(const char* key) { return find(std::string(key)); }
    const_iterator find(const char* key) const { return find(std::string(key)); }

    bool contains(const std::string& key) const { return find(key) != end(); }
    bool contains(const char* key) const { return find(key) != end(); }
    size_t count(const std::string& key) const { return contains(key) ? 1 : 0; }
    size_t count(const char* key) const { return contains(key) ? 1 : 0; }

    Json& at(const std::string& key) {
        auto it = find(key);
        if (it == end()) fail("json: key not found");
        return *it;
    }
    const Json& at(const std::string& key) const {
        auto it = find(key);
        if (it == end()) fail("json: key not found");
        return *it;
    }
    Json& at(const char* key) { return at(std::string(key)); }
    const Json& at(const char* key) const { return at(std::string(key)); }
    Json& at(size_t idx) {
        if (!is_array()) fail_type("array");
        if (idx >= arr().size()) fail("json: array index out of range");
        return arr()[idx];
    }
    const Json& at(size_t idx) const {
        if (!is_array()) fail_type("array");
        if (idx >= arr().size()) fail("json: array index out of range");
        return arr()[idx];
    }
    Json& at(int idx) { return at((size_t)idx); }
    const Json& at(int idx) const { return at((size_t)idx); }

    Json& operator[](const std::string& key) {
        if (is_null()) v_ = object_t();
        if (!is_object()) fail_type("object");
        auto it = find(key);
        if (it != end()) return *it;
        obj().emplace_back(key, Json());
        return obj().back().second;
    }
    Json& operator[](const char* key) { return operator[](std::string(key)); }
    Json& operator[](std::string_view key) { return operator[](std::string(key)); }
    const Json& operator[](const std::string& key) const { return at(key); }
    const Json& operator[](const char* key) const { return at(key); }
    const Json& operator[](std::string_view key) const { return at(std::string(key)); }

    Json& operator[](size_t idx) {
        if (is_null()) v_ = array_t();
        if (!is_array()) fail_type("array");
        if (idx >= arr().size()) arr().resize(idx + 1);
        return arr()[idx];
    }
    Json& operator[](int idx) { return operator[]((size_t)idx); }
    Json& operator[](int64_t idx) { return operator[]((size_t)idx); }
    const Json& operator[](size_t idx) const { return at(idx); }
    const Json& operator[](int idx) const { return at((size_t)idx); }
    const Json& operator[](int64_t idx) const { return at((size_t)idx); }

    Json& front() {
        if (is_array() || is_object()) {
            if (empty()) fail("json: front() on empty container");
            return *begin();
        }
        return *this;
    }
    const Json& front() const {
        if (is_array() || is_object()) {
            if (empty()) fail("json: front() on empty container");
            return *begin();
        }
        return *this;
    }
    Json& back() {
        if (is_array()) {
            if (arr().empty()) fail("json: back() on empty array");
            return arr().back();
        }
        if (is_object()) {
            if (obj().empty()) fail("json: back() on empty object");
            return obj().back().second;
        }
        return *this;
    }
    const Json& back() const {
        if (is_array()) {
            if (arr().empty()) fail("json: back() on empty array");
            return arr().back();
        }
        if (is_object()) {
            if (obj().empty()) fail("json: back() on empty object");
            return obj().back().second;
        }
        return *this;
    }

    /* ------------------------------------------------------------------ conversion
     * the predecessor's rules: numbers convert between number kinds; nothing converts across the
     * string/bool/number/container boundaries without throwing. */
    template <class T>
    T get() const {
        if constexpr (std::is_same_v<T, Json>) {
            return *this;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (!is_boolean()) fail_type("boolean");
            return std::get<bool>(v_);
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (!is_string()) fail_type("string");
            return std::get<string_t>(v_);
        } else if constexpr (std::is_same_v<T, std::string_view>) {
            if (!is_string()) fail_type("string");
            return std::string_view(std::get<string_t>(v_));
        } else if constexpr (std::is_arithmetic_v<T>) {
            /* the predecessor's arithmetic read also accepts a boolean, as 0/1. */
            if (const auto* p = std::get_if<bool>(&v_)) return (T)(*p ? 1 : 0);
            if (!is_number()) fail_type("number");
            if (const auto* p = std::get_if<int64_t>(&v_)) return (T)*p;
            if (const auto* p = std::get_if<uint64_t>(&v_)) return (T)*p;
            return (T)std::get<double>(v_);
        } else {
            return get_container<T>();
        }
    }

private:
    template <class T>
    T get_container() const {
        /* vector<U>: element-wise. map<string,U>: member-wise. Everything else fails like
         * the predecessor's default serializer would. */
        if constexpr (requires { typename T::key_type; typename T::mapped_type; }) {
            if (!is_object()) fail_type("object");
            T out;
            for (const auto& kv : obj()) out.emplace(kv.first, kv.second.get<typename T::mapped_type>());
            return out;
        } else if constexpr (requires { typename T::value_type; }) {
            if (!is_array()) fail_type("array");
            T out;
            for (const Json& e : arr()) {
                if constexpr (requires(T& c, typename T::value_type v) { c.push_back(v); })
                    out.push_back(e.get<typename T::value_type>());
                else
                    out.insert(e.get<typename T::value_type>());
            }
            return out;
        } else {
            fail("json: unsupported get<T>() target");
        }
    }

public:
    /* A JSON Pointer (RFC 6901), for the value()/at() lookups spelled through one. */
    class json_pointer {
        std::string p_;

    public:
        json_pointer() = default;
        explicit json_pointer(std::string p) : p_(std::move(p)) {}
        json_pointer(const char* p) : p_(p) {}
        const std::string& to_string() const { return p_; }
        json_pointer operator/(const std::string& seg) const {
            std::string e;
            for (char c : seg) {
                if (c == '~') e += "~0";
                else if (c == '/') e += "~1";
                else e += c;
            }
            return json_pointer(p_ + "/" + e);
        }
        json_pointer operator/(size_t idx) const { return json_pointer(p_ + "/" + std::to_string(idx)); }
    };

private:
    /* Walk a pointer; returns nullptr when any hop misses (at() turns that into a throw). */
    const Json* ptr_at(const json_pointer& jp) const {
        const Json* cur = this;
        const std::string& p = jp.to_string();
        size_t i = 0;
        while (i < p.size()) {
            if (p[i] != '/') return nullptr;
            const size_t next = p.find('/', i + 1);
            std::string tok = p.substr(i + 1, next == std::string::npos ? next : next - i - 1);
            for (size_t k = 0; k < tok.size(); ++k) {
                if (tok[k] == '~' && k + 1 < tok.size()) {
                    tok.replace(k, 2, tok[k + 1] == '0' ? "~" : tok[k + 1] == '1' ? "/" : "~");
                }
            }
            if (cur->is_object()) {
                auto it = cur->find(tok);
                if (it == cur->end()) return nullptr;
                cur = &*it;
            } else if (cur->is_array()) {
                char* endp = nullptr;
                const unsigned long long idx = std::strtoull(tok.c_str(), &endp, 10);
                if (!endp || *endp != '\0' || tok.empty() || idx >= cur->size()) return nullptr;
                cur = &cur->at((size_t)idx);
            } else {
                return nullptr;
            }
            i = next == std::string::npos ? p.size() : next;
        }
        return cur;
    }

public:
    const Json& at(const json_pointer& jp) const {
        const Json* r = ptr_at(jp);
        if (!r) fail("json: json_pointer does not resolve");
        return *r;
    }
    bool contains(const json_pointer& jp) const { return ptr_at(jp) != nullptr; }
    template <class T>
    T value(const json_pointer& jp, const T& def) const {
        const Json* r = ptr_at(jp);
        return r ? r->get<T>() : def;
    }

    /* the predecessor's get_to: assign the converted value into an existing object. */
    template <class T>
    void get_to(T& out) const {
        out = get<T>();
    }

    template <class T>
    const std::string& get_ref() const {
        static_assert(std::is_same_v<std::decay_t<T>, std::string>,
                      "get_ref is only defined for strings");
        if (!is_string()) fail_type("string");
        return std::get<string_t>(v_);
    }
    template <class T>
    std::string& get_ref() {
        static_assert(std::is_same_v<std::decay_t<T>, std::string>,
                      "get_ref is only defined for strings");
        if (!is_string()) fail_type("string");
        return std::get<string_t>(v_);
    }

    template <class T>
    T value(const std::string& key, const T& def) const {
        auto it = find(key);
        return it == end() ? def : it->get<T>();
    }
    template <class T>
    T value(const char* key, const T& def) const {
        return value(std::string(key), def);
    }
    /* the predecessor's value() with a const char* default yields a string. */
    std::string value(const std::string& key, const char* def) const {
        return value<std::string>(key, std::string(def));
    }
    std::string value(const char* key, const char* def) const {
        return value(std::string(key), def);
    }

    /* The implicit conversions the predecessor allowed (assignment from Json to string/numeric/bool
     * targets is all over the chat lift). Constrained to those kinds so overloads do not go
     * ambiguous anywhere a Json merely sits in a boolean or numeric context by accident.
     * The string conversion is a plain overload so it wins over the arithmetic template
     * (std::string's assignment from char would otherwise tie with it). Character types are
     * excluded from the arithmetic route for the same reason. */
    operator std::string() const { return get<std::string>(); }
    template <class T,
              std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, char> &&
                                   !std::is_same_v<T, signed char> &&
                                   !std::is_same_v<T, unsigned char> && !std::is_same_v<T, wchar_t>,
                               int> = 0>
    operator T() const {
        return get<T>();
    }
    /* Containers convert too (the predecessor's value semantics: vector<float> from an array).
     * Restricted to the sequence/associative containers get<>() supports -- an unconstrained
     * value_type match would also claim optional<T> and shadow its converting constructor. */
    template <class T, std::enable_if_t<json_detail::is_gettable_container<T>::value, int> = 0>
    operator T() const {
        return get<T>();
    }

    /* ------------------------------------------------------------------ mutation */
    void push_back(const Json& v) {
        if (is_null()) v_ = array_t();
        if (!is_array()) fail_type("array");
        arr().push_back(v);
    }
    void push_back(Json&& v) {
        if (is_null()) v_ = array_t();
        if (!is_array()) fail_type("array");
        arr().push_back(std::move(v));
    }
    template <class... Args>
    Json& emplace_back(Args&&... args) {
        if (is_null()) v_ = array_t();
        if (!is_array()) fail_type("array");
        return arr().emplace_back(std::forward<Args>(args)...);
    }

    size_t erase(const std::string& key) {
        if (!is_object()) return 0;
        auto& o = obj();
        for (auto it = o.begin(); it != o.end(); ++it)
            if (it->first == key) {
                o.erase(it);
                return 1;
            }
        return 0;
    }
    size_t erase(const char* key) { return erase(std::string(key)); }
    iterator erase(iterator pos) {
        if (is_array()) return iterator::for_array(this, arr().erase(pos.a_));
        if (is_object()) return iterator::for_object(this, obj().erase(pos.o_));
        fail("json: erase() on a scalar");
    }
    iterator erase(size_t idx) {
        if (!is_array()) fail_type("array");
        if (idx >= arr().size()) fail("json: array index out of range");
        return iterator::for_array(this, arr().erase(arr().begin() + (difference_type)idx));
    }
    iterator insert(iterator pos, const Json& v) {
        if (!is_array()) fail_type("array");
        return iterator::for_array(this, arr().insert(pos.a_, v));
    }
    template <class It>
    iterator insert(iterator pos, It first, It last) {
        if (!is_array()) fail_type("array");
        return iterator::for_array(this, arr().insert(pos.a_, first, last));
    }

    /* the predecessor's update: copy other's members over this object's, one level deep; with the
     * merge flag, nested objects merge instead of replacing. */
    void update(const Json& other, bool merge_objects = false) {
        if (!other.is_object()) fail_type("object");
        if (is_null()) v_ = object_t();
        if (!is_object()) fail_type("object");
        for (const auto& kv : other.obj()) {
            if (merge_objects) {
                auto it = find(kv.first);
                if (it != end() && it->is_object() && kv.second.is_object()) {
                    it->update(kv.second, true);
                    continue;
                }
            }
            operator[](kv.first) = kv.second;
        }
    }

    /* ------------------------------------------------------------------ comparison */
    bool operator==(const Json& o) const {
        if (is_number_kind() && o.is_number_kind()) {
            /* Exact cross-kind numeric equality: integers compare exactly; a float compares
             * equal only to integers it represents exactly (long double carries both). */
            return as_long_double() == o.as_long_double();
        }
        if (type() != o.type()) return false;
        switch (type()) {
        case value_t::null: return true;
        case value_t::boolean: return std::get<bool>(v_) == std::get<bool>(o.v_);
        case value_t::string: return std::get<string_t>(v_) == std::get<string_t>(o.v_);
        case value_t::array: return arr() == o.arr();
        /* Members in order, as ordered_json compared them: key order is part of what the
         * server writes and a template renders, so two objects that differ in it differ. */
        case value_t::object: return obj() == o.obj();
        default: return false;
        }
    }
    bool operator!=(const Json& o) const { return !(*this == o); }
    bool operator==(std::nullptr_t) const { return is_null(); }
    bool operator!=(std::nullptr_t) const { return !is_null(); }
    bool operator==(const char* s) const { return is_string() && std::get<string_t>(v_) == s; }
    bool operator!=(const char* s) const { return !(*this == s); }
    bool operator==(const std::string& s) const {
        return is_string() && std::get<string_t>(v_) == s;
    }
    bool operator!=(const std::string& s) const { return !(*this == s); }
    bool operator==(bool b) const { return is_boolean() && std::get<bool>(v_) == b; }
    bool operator!=(bool b) const { return !(*this == b); }
    template <class T,
              std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, int> = 0>
    bool operator==(T v) const {
        return is_number_kind() && as_long_double() == (long double)v;
    }
    template <class T,
              std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, int> = 0>
    bool operator!=(T v) const {
        return !(*this == v);
    }

    /* ------------------------------------------------------------------ text (json.cpp) */
    std::string dump(int indent = -1, char indent_char = ' ', bool ensure_ascii = false,
                     error_handler_t eh = error_handler_t::strict) const;

    static Json parse(std::string_view text);
    static Json parse(std::istream& in);
    static Json parse(std::istream& in, std::nullptr_t, bool allow_exceptions,
                      bool ignore_comments = false);
    static Json parse(const char* text) { return parse(std::string_view(text)); }
    static Json parse(const std::string& text) { return parse(std::string_view(text)); }
    static Json parse(std::string_view text, std::nullptr_t, bool allow_exceptions,
                      bool ignore_comments = false);
    static Json parse(const std::string& text, std::nullptr_t cb, bool allow_exceptions,
                      bool ignore_comments = false) {
        return parse(std::string_view(text), cb, allow_exceptions, ignore_comments);
    }
    static Json parse(const char* text, std::nullptr_t cb, bool allow_exceptions,
                      bool ignore_comments = false) {
        return parse(std::string_view(text), cb, allow_exceptions, ignore_comments);
    }
    static bool accept(std::string_view text, bool ignore_comments = false);

    /* The bounded read core/server's parse_json_bounded is built on: at most `budget`
     * values (containers count once, keys not at all), nothing nested deeper than
     * max_depth below the root (root depth 0). Sets *over/*deep and returns null instead
     * of a tree when a bound trips; returns discarded on a syntax error. */
    static Json parse_bounded(std::string_view text, int64_t budget, int max_depth, bool* over,
                              bool* deep);

private:
    using difference_type = std::ptrdiff_t;
};

inline void Json::fail_type(const char* want) const {
    std::string msg = "json: value is ";
    msg += type_name();
    msg += ", not ";
    msg += want;
    throw exception(msg);
}

/* d appended as Json::dump writes a float: the shortest digits that read back as d, fixed
 * notation while the decimal point is within (-4, 15] places of the first digit (with ".0" on
 * an integral value), else d.ddde+XX; a non-finite d as null. For writers that emit JSON
 * numbers without building a tree. */
void json_append_double(std::string& out, double d);

/* f appended in the same layout with the shortest digits that read back as f as a float: what a
 * float32 holds, without the digits widening it to a double would add (0.1f is 0.1, not
 * 0.10000000149011612). */
void json_append_float(std::string& out, float f);

std::istream& operator>>(std::istream& in, Json& j);
std::ostream& operator<<(std::ostream& out, const Json& j);

}  /* namespace rad */

template <>
struct std::tuple_size<rad::Json::items_proxy<false>::proxy>
    : std::integral_constant<size_t, 2> {};
template <>
struct std::tuple_size<rad::Json::items_proxy<true>::proxy>
    : std::integral_constant<size_t, 2> {};
template <>
struct std::tuple_element<0, rad::Json::items_proxy<false>::proxy> {
    using type = std::string;
};
template <>
struct std::tuple_element<0, rad::Json::items_proxy<true>::proxy> {
    using type = std::string;
};
template <>
struct std::tuple_element<1, rad::Json::items_proxy<false>::proxy> {
    using type = rad::Json&;
};
template <>
struct std::tuple_element<1, rad::Json::items_proxy<true>::proxy> {
    using type = const rad::Json&;
};
