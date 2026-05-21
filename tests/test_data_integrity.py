"""Tests for data integrity — ensures the knowledge graph accurately represents the source data."""

import re


class TestFacultyCount:
    """Verify the exact faculty count matches the authoritative source."""

    def test_exactly_24_faculty_nodes(self, graph, canonical_faculty):
        """The graph must have exactly 24 Faculty nodes (matching ee_faculty-list.html.md)."""
        faculty_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'Faculty']
        assert len(faculty_nodes) == 24, (
            f"Expected 24 Faculty nodes, got {len(faculty_nodes)}: {sorted(faculty_nodes)}"
        )

    def test_all_canonical_names_present(self, graph, canonical_faculty):
        """Every one of the 24 known faculty names must exist as a graph node."""
        faculty_nodes = {n for n, d in graph.nodes(data=True) if d.get('label') == 'Faculty'}
        missing = []
        for name in canonical_faculty:
            if name not in faculty_nodes:
                missing.append(name)
        assert not missing, f"Missing faculty nodes: {missing}"

    def test_no_external_persons_labeled_as_faculty(self, graph, canonical_faculty):
        """No ExternalPerson should be labeled as Faculty."""
        faculty_nodes = [n for n, d in graph.nodes(data=True) if d.get('label') == 'Faculty']
        canonical_set = set(canonical_faculty)
        imposters = [n for n in faculty_nodes if n not in canonical_set]
        assert not imposters, f"Non-canonical names incorrectly labeled as Faculty: {imposters}"


class TestFacultyAttributes:
    """Verify each faculty node has essential attributes."""

    def test_all_faculty_have_email(self, graph, canonical_faculty):
        """Every canonical faculty should have an email address."""
        missing_email = []
        for name in canonical_faculty:
            if graph.has_node(name):
                email = graph.nodes[name].get('email', '')
                if not email:
                    missing_email.append(name)
        # Some faculty may not have email in the data, so warn rather than fail
        assert len(missing_email) <= 5, f"Too many faculty without email: {missing_email}"

    def test_all_faculty_have_designation(self, graph, canonical_faculty):
        """Every canonical faculty should have a designation."""
        missing = []
        for name in canonical_faculty:
            if graph.has_node(name):
                desg = graph.nodes[name].get('designation', '')
                if not desg:
                    missing.append(name)
        assert len(missing) <= 5, f"Too many faculty without designation: {missing}"

    def test_faculty_connected_to_department(self, graph, canonical_faculty):
        """Every faculty should have a MEMBER_OF edge to the department."""
        dept_id = "IIT Jammu EE Department"
        not_connected = []
        for name in canonical_faculty:
            if graph.has_node(name):
                if not graph.has_edge(name, dept_id):
                    not_connected.append(name)
        assert not not_connected, f"Faculty not connected to department: {not_connected}"


class TestDepartmentNode:
    """Verify the department node has correct metadata."""

    def test_department_exists(self, graph):
        assert graph.has_node("IIT Jammu EE Department")

    def test_department_faculty_count(self, graph):
        """Department node should store the correct faculty count."""
        dept_data = graph.nodes["IIT Jammu EE Department"]
        count = dept_data.get('faculty_count', 0)
        assert count == 24, f"Department faculty_count is {count}, expected 24"


class TestPhDStudents:
    """Verify PhD students are correctly parsed."""

    def test_phd_students_exist(self, graph):
        students = [n for n, d in graph.nodes(data=True) if d.get('label') == 'PhDStudent']
        assert len(students) > 0, "No PhD students found in graph"

    def test_exactly_66_phd_students(self, graph):
        """The graph should reflect the full current PhD roster page."""
        students = [n for n, d in graph.nodes(data=True) if d.get('label') == 'PhDStudent']
        assert len(students) == 66, f"Expected 66 PhD students, got {len(students)}"

    def test_phd_students_have_supervisors(self, graph):
        """At least some PhD students should have SUPERVISED_BY edges."""
        students = [n for n, d in graph.nodes(data=True) if d.get('label') == 'PhDStudent']
        with_supervisor = 0
        for s in students:
            for _, target, data in graph.out_edges(s, data=True):
                if data.get('type') == 'SUPERVISED_BY':
                    with_supervisor += 1
                    break
        assert with_supervisor > 0, "No PhD students have supervisors"

    def test_department_phd_count(self, graph):
        """Department node should store the exact PhD scholar count."""
        dept_data = graph.nodes["IIT Jammu EE Department"]
        count = dept_data.get('phd_student_count', 0)
        assert count == 66, f"Department phd_student_count is {count}, expected 66"


class TestExternalPersons:
    """Verify external collaborators are correctly separated from faculty."""

    def test_external_persons_exist(self, graph):
        """There should be ExternalPerson nodes for non-IIT-Jammu supervisors."""
        externals = [n for n, d in graph.nodes(data=True) if d.get('label') == 'ExternalPerson']
        assert len(externals) > 0, "No ExternalPerson nodes found"

    def test_external_persons_not_in_department(self, graph):
        """ExternalPerson nodes should NOT have MEMBER_OF edges to the department."""
        dept_id = "IIT Jammu EE Department"
        externals = [n for n, d in graph.nodes(data=True) if d.get('label') == 'ExternalPerson']
        connected = [n for n in externals if graph.has_edge(n, dept_id)]
        assert not connected, f"External persons incorrectly connected to department: {connected}"
