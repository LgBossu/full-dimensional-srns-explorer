import importlib
import inspect
import pydoc
import re
import types


def extract_full_docs(module_name, output_file="module_docs.txt", max_depth=2):
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Could not import module '{module_name}'")
        return

    visited = set()

    def write_header(f, name, level):
        header = f"\n{'=' * (80 - 2*level)}\n{name}\n{'=' * (80 - 2*level)}\n"
        f.write(header)

    def walk(obj, name, f, level=0):
        if id(obj) in visited or level > max_depth:
            return
        visited.add(id(obj))

        write_header(f, name, level)
        doc = pydoc.render_doc(obj) or "(No docstring found)"
        f.write(re.sub(r".\x08", "", doc) + "\n\n")

        if isinstance(obj, types.ModuleType) or inspect.isclass(obj):
            try:
                for member_name, member in inspect.getmembers(obj):
                    # Skip builtins, private, or huge base objects
                    if member_name.startswith("__") and member_name.endswith("__"):
                        continue
                    # Don't go too deep into imported modules
                    if (
                        inspect.ismodule(member)
                        and member.__name__.split(".")[0] != module_name.split(".")[0]
                    ):
                        continue
                    full_name = f"{name}.{member_name}"
                    walk(member, full_name, f, level + 1)
            except Exception as e:
                f.write(f"\n[Error while inspecting {name}: {e}]\n")

    with open(output_file, "w", encoding="utf-8") as f:
        walk(module, module_name, f)

    print(f"Documentation safely written to '{output_file}' ✅")


# Example usage:
if __name__ == "__main__":
    name = "polytopewalk"
    extension = ".txt"
    extract_full_docs(name, output_file=f"{name}_doc{extension}", max_depth=5)
