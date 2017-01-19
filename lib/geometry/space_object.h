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

#ifndef MF_SPACE_OBJECT_H_
#define MF_SPACE_OBJECT_H_

#include "pose.h"
#include "angle.h"
#include "../eigen.h"
#include <set>
#include <string>
#include <mutex>

namespace mf {

struct bounding_box;

/// Object that has a pose in 3D space relative to a coordinate system, base class.
/** Can have a parent space object, meaning its pose will be defined relative to that parent.
 ** No parent means pose is absolute. Parent can be reassigned. Keeps references to its children,
 ** Guarantees that pointers to parent or children get properly removed when either child or parent gets deleted.*/
class space_object {
private:
	space_object* parent_;
	std::set<space_object*> children_;
	pose pose_;
	
	void transform_(const Eigen_affine3&);
	
	void detach_from_parent_();
	void detach_from_children_();
	void attach_to_parent_();

protected:
	virtual void do_update_pose() { }
		
public:
	explicit space_object(const pose& = pose());
	space_object(const pose&, space_object& par);

	space_object(const space_object&);
	space_object& operator=(const space_object&);
	virtual ~space_object();

	bool has_parent_space_object() const;
	const space_object& parent_space_object() const;
	space_object& parent_space_object();
	
	const pose& relative_pose() const;
	pose absolute_pose() const;
	
	void set_relative_pose(const pose&);
	void set_no_relative_pose() { set_relative_pose(pose()); }
	
	Eigen_affine3 transformation_from(const space_object&) const;
	Eigen_affine3 transformation_to(const space_object&) const;
	
	void set_parent(space_object&, const pose& new_relative_pose = pose());
	void set_no_parent(const pose& new_pose = pose());
	
	void make_sibling(const space_object&, const pose& new_pose = pose());
				
	template<typename Transformation>
	void transform(const Transformation& t) {
		transform_(Eigen_affine3(t));
	}
	
	template<typename Transformation>
	void transform(const Transformation& rt, const space_object& relative_to) {
		Eigen_affine3 btw = relative_to.absolute_pose().transformation_to_world();
		Eigen_affine3 t = btw * rt * btw.inverse();
		transform_(t);
	}
	
	void move(const Eigen_vec3& t) { transform(Eigen_translation3(t)); }
	void move(float x, float y, float z) { move(Eigen_vec3(x, y, z)); }
	void move_x(float x) { move(x, 0, 0); }
	void move_y(float y) { move(0, y, 0); }
	void move_z(float z) { move(0, 0, z); }
	
	void rotate_x_axis(angle a, const Eigen_vec3& c = Eigen_vec3::Zero());
	void rotate_y_axis(angle a, const Eigen_vec3& c = Eigen_vec3::Zero());
	void rotate_z_axis(angle a, const Eigen_vec3& c = Eigen_vec3::Zero());
	
	void look_at(const space_object&);
	
	virtual bounding_box box() const;
};


inline Eigen_affine3 pose_transformation(const space_object& from, const space_object& to) {
	return from.transformation_to(to);
}



}

#endif
