// Copyright (c) 2021-2022 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//shared_ptrを要素に持つmapとvectorについて条件に合致したもののみを抽出しつつweak_ptrとして走査するイテレータ
#pragma once
#include "Common.h"
#include <memory>
#include <iterator>
#include <vector>
#include <map>
#include <functional>
#include <pybind11/pybind11.h>
namespace py=pybind11;

template<class T,class U>
class MapIterable;
template<class T,class U>
class PYBIND11_EXPORT MapIterator:public std::iterator<std::forward_iterator_tag,std::weak_ptr<T>>{
    public:
    typedef const typename std::map<std::string,std::shared_ptr<U>> ContainerType;
    typedef typename ContainerType::const_iterator ItType;
    typedef typename std::shared_ptr<const T> ConstSharedType;
    typedef typename std::weak_ptr<T> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    MapIterator(ItType it_,MatcherType matcher_,ItType end_):it(it_),matcher(matcher_),end(end_){}
    MapIterator(const MapIterator<T,U>& other):it(other.it),matcher(other.matcher),end(other.end){}
    ~MapIterator(){}
    bool operator==(const MapIterator<T,U>& other){
        return it==other.it;
    }
    bool operator!=(const MapIterator<T,U>& other){
        return it!=other.it;
    }
    const WeakType operator*() const{
        if constexpr (std::is_same_v<T,U>){
            return (*it).second;
        }else{
            return std::dynamic_pointer_cast<T>((*it).second);
        }
    }
    WeakType operator*(){
        if constexpr (std::is_same_v<T,U>){
            return (*it).second;
        }else{
            return std::dynamic_pointer_cast<T>((*it).second);
        }
    }
    const WeakType operator->() const{
        if constexpr (std::is_same_v<T,U>){
            return it->second;
        }else{
            return std::dynamic_pointer_cast<T>(it->second);
        }
    }
    WeakType operator->(){
        if constexpr (std::is_same_v<T,U>){
            return it->second;
        }else{
            return std::dynamic_pointer_cast<T>(it->second);
        }
    }
    MapIterator<T,U>& operator++(){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<T,U>){
                if(it==end || matcher(it->second)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<T>(it->second) && matcher(std::dynamic_pointer_cast<T>(it->second)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    MapIterator<T,U> operator++(int){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<T,U>){
                if(it==end || matcher(it->second)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<T>(it->second) && matcher(std::dynamic_pointer_cast<T>(it->second)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    private:
    MatcherType matcher;
    ItType it,end;
};
template<class T,class U>
class MapIterable{
    public:
    typedef const typename std::map<std::string,std::shared_ptr<U>> ContainerType;
    typedef MapIterator<T,U> iterator;
    typedef typename std::shared_ptr<const T> ConstSharedType;
    typedef typename std::weak_ptr<T> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    MapIterable(ContainerType& container_,MatcherType matcher_):container(container_),matcher(matcher_),it(begin()){}
    ~MapIterable(){}
    MapIterable<T,U>& iter(){
        it=begin();
        first=true;
        return *this;
    }
    WeakType next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==end()){
            throw pybind11::stop_iteration();
        }
        return *it;
    }
    iterator begin(){
        auto ret=container.begin();
        while(ret!=container.end()){
            if constexpr (std::is_same_v<T,U>){
                if(matcher(ret->second)){
                    return iterator(ret,matcher,container.end());
                }
            }else{
                if(util::isinstance<T>(ret->second) && matcher(std::dynamic_pointer_cast<T>(ret->second))){
                    return iterator(ret,matcher,container.end());
                }
            }
            ++ret;
        }
        return iterator(ret,matcher,container.end());
    }
    iterator end(){
        return iterator(container.end(),matcher,container.end());
    }
    private:
    MatcherType matcher;
    ContainerType& container;
    iterator it;
    bool first=true;
};
template<class T,class U>
class VectorIterable;
template<class T,class U>
class VectorIterator:public std::iterator<std::forward_iterator_tag,std::weak_ptr<T>>{
    public:
    typedef const typename std::vector<std::shared_ptr<U>> ContainerType;
    typedef typename ContainerType::const_iterator ItType;
    typedef typename std::shared_ptr<const T> ConstSharedType;
    typedef typename std::weak_ptr<T> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    VectorIterator(ItType it_,MatcherType matcher_,ItType end_):it(it_),matcher(matcher_),end(end_){}
    VectorIterator(const VectorIterator<T,U>& other):it(other.it),matcher(other.matcher),end(other.end){}
    ~VectorIterator(){}
    bool operator==(const VectorIterator<T,U>& other){
        return it==other.it;
    }
    bool operator!=(const VectorIterator<T,U>& other){
        return it!=other.it;
    }
    const WeakType operator*() const{
        if constexpr (std::is_same_v<T,U>){
            return *it;
        }else{
            return std::dynamic_pointer_cast<T>(*it);
        }
    }
    const WeakType operator->() const{
        if constexpr (std::is_same_v<T,U>){
            return it->operator();
        }else{
            return std::dynamic_pointer_cast<T>(it->operator());
        }
    }
    VectorIterator<T,U>& operator++(){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<T,U>){
                if(it==end || matcher(*it)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<T>(*it) && matcher(std::dynamic_pointer_cast<T>(*it)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    VectorIterator<T,U> operator++(int){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<T,U>){
                if(it==end || matcher(*it)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<T>(*it) && matcher(std::dynamic_pointer_cast<T>(*it)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    private:
    MatcherType matcher;
    ItType it,end;
};
template<class T,class U>
class VectorIterable{
    public:
    typedef const typename std::vector<std::shared_ptr<U>> ContainerType;
    typedef VectorIterator<T,U> iterator;
    //typedef const VectorIterator<T,U> const_iterator;
    typedef typename std::shared_ptr<const T> ConstSharedType;
    typedef typename std::weak_ptr<T> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    VectorIterable(ContainerType& container_,MatcherType matcher_):container(container_),matcher(matcher_),it(begin()){}
    ~VectorIterable(){}
    VectorIterable<T,U>& iter(){
        //std::cout<<"iter()"<<std::endl;
        it=begin();
        first=true;
        return *this;
    }
    WeakType next(){
        //std::cout<<"next()"<<std::endl;
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==end()){
            throw pybind11::stop_iteration();
        }
        return *it;
    }
    iterator begin(){
        auto ret=container.begin();
        while(ret!=container.end()){
            if constexpr (std::is_same_v<T,U>){
                if(matcher(*ret)){
                    return iterator(ret,matcher,container.end());
                }
            }else{
                if(util::isinstance<T>(*ret) && matcher(std::dynamic_pointer_cast<T>(*ret))){
                    return iterator(ret,matcher,container.end());
                }
            }
            ++ret;
        }
        return iterator(ret,matcher,container.end());
    }
    iterator end(){
        return iterator(container.end(),matcher,container.end());
    }
    private:
    MatcherType matcher;
    ContainerType& container;
    iterator it;
    bool first=true;
};
