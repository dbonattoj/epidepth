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

#include "space_object.h"
#include "bounding_box.h"
#include "spherical_coordinates.h"
#include <stdexcept>
#include <mutex>

namespace mf {

space_object::space_object(const pose& ps, space_object& par) :
parent_(&par), pose_(ps) {
	attach_to_parent_();
}


space_object::space_object(const pose& ps) :
parent_(nullptr), pose_(ps) { }


space_object::space_object(const space_object& obj) :
parent_(obj.parent_), pose_(obj.pose_) {
	attach_to_parent_();
	// don't copy children_ array: these children are not children of this new space_object
}

space_object& space_object::operator=(const space_object& obj) {
	if(this == &obj) return *this;
	detach_from_parent_();
	detach_from_children_();
	parent_ = obj.parent_;
	children_.clear();
	pose_ = obj.pose_;
	attach_to_parent_();
	return *this;
}


space_object::~space_object() {
	detach_from_parent_();
	detach_from_children_();
}


void space_object::detach_from_parent_() {
	if(parent_) parent_->children_.erase(this);
}

void space_object::detach_from_children_() {
	for(auto child : children_)
		if(child->parent_ == this) child->parent_ = nullptr;
}


void space_object::attach_to_parent_() {
	if(parent_) parent_->children_.insert(this);
}


bool space_object::has_parent_space_object() const {
	return (parent_ != nullptr);
}


const space_object& space_object::parent_space_object() const {
	if(! has_parent_space_object()) throw std::logic_error("space object has no parent");
	else return *parent_;
}


space_object& space_object::parent_space_object() {
	if(! has_parent_space_object()) throw std::logic_error("space object has no parent");
	else return *parent_;
}

const pose& space_object::relative_pose() const {
	return pose_;
}


pose space_object::absolute_pose() const {
	Eigen_affine3 trans = pose_.transformation_to_world();
	if(has_parent_space_object())
		trans = parent_space_object().absolute_pose().transformation_to_world() * trans;
	return pose(trans);
}


void space_object::set_relative_pose(const pose& ps) {
	pose_ = ps;
	this->do_update_pose();
}


void space_object::set_parent(space_object& par, const pose& new_relative_pose) {
	detach_from_parent_();
	parent_ = &par;
	attach_to_parent_();
	pose_ = new_relative_pose;
	this->do_update_pose();
}


void space_object::set_no_parent(const pose& new_pose) {
	detach_from_parent_();
	pose_ = new_pose;
	this->do_update_pose();
}


void space_object::make_sibling(const space_object& obj, const pose& new_pose) {
	attach_to_parent_();
	parent_ = obj.parent_;
	pose_ = new_pose;
	detach_from_parent_();
	this->do_update_pose();
}


void space_object::transform_(const Eigen_affine3& t) {
	Eigen_affine3 pose_transformation = pose_.transformation_to_world();
	pose_transformation = t * pose_transformation;
	pose_ = pose(pose_transformation);
	this->do_update_pose();
}


bounding_box space_object::box() const {
	// default implementation
	throw std::logic_error("space object has no bounding box");
}


Eigen_affine3 space_object::transformation_to(const space_object& obj) const {
	Eigen_affine3 obj_to_world = obj.absolute_pose().transformation_to_world();
	Eigen_affine3 world_to_this = absolute_pose().transformation_from_world();
	return world_to_this * obj_to_world;
}


Eigen_affine3 space_object::transformation_from(const space_object& obj) const {
	return obj.transformation_to(*this);
}


void space_object::rotate_x_axis(angle a, const Eigen_vec3& c) {
	Eigen_translation3 t(c + pose_.position);
	transform(t * Eigen_angleaxis(a, Eigen_vec3::UnitX()) * t.inverse());
}


void space_object::rotate_y_axis(angle a, const Eigen_vec3& c) {
	Eigen_translation3 t(c + pose_.position);
	transform(t * Eigen_angleaxis(a, Eigen_vec3::UnitY()) * t.inverse());
}


void space_object::rotate_z_axis(angle a, const Eigen_vec3& c) {
	Eigen_translation3 t(c + pose_.position);
	transform(t * Eigen_angleaxis(a, Eigen_vec3::UnitZ()) * t.inverse());
}



void space_object::look_at(const space_object& obj) {
	Eigen_vec3 at_obj = transformation_from(obj) * Eigen_vec3::Zero();
	Eigen_vec3 at_depth(0, 0, 1);
	pose_.orientation.setFromTwoVectors(at_depth, at_obj);
	pose_.orientation.normalize();
}


}
