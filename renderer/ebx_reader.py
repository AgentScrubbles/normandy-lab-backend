"""
Schema-less EBX (Frostbite blueprint) reader for DA:I.

Parses EBX Version2 files without needing the TypeLibrary/SDK DLLs.
Produces a generic JSON-compatible tree structure.
Based on FrostyToolsuite's EbxReader.cs.
"""

import struct
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional


# --- EBX field type enum ---

class EbxFieldType:
    INHERITED = 0x00
    DB_OBJECT = 0x01
    STRUCT = 0x02
    POINTER = 0x03
    ARRAY = 0x04
    STRING = 0x06
    CSTRING = 0x07
    ENUM = 0x08
    FILE_REF = 0x09
    BOOLEAN = 0x0A
    INT8 = 0x0B
    UINT8 = 0x0C
    INT16 = 0x0D
    UINT16 = 0x0E
    INT32 = 0x0F
    UINT32 = 0x10
    UINT64 = 0x11
    INT64 = 0x12
    FLOAT32 = 0x13
    FLOAT64 = 0x14
    GUID = 0x15
    SHA1 = 0x16
    RESOURCE_REF = 0x17
    DELEGATE = 0x18
    TYPE_REF = 0x19
    BOXED_VALUE_REF = 0x1A

    _NAMES = {
        0x00: "Inherited", 0x01: "DbObject", 0x02: "Struct", 0x03: "Pointer",
        0x04: "Array", 0x06: "String", 0x07: "CString", 0x08: "Enum",
        0x09: "FileRef", 0x0A: "Boolean", 0x0B: "Int8", 0x0C: "UInt8",
        0x0D: "Int16", 0x0E: "UInt16", 0x0F: "Int32", 0x10: "UInt32",
        0x11: "UInt64", 0x12: "Int64", 0x13: "Float32", 0x14: "Float64",
        0x15: "Guid", 0x16: "Sha1", 0x17: "ResourceRef", 0x18: "Delegate",
        0x19: "TypeRef", 0x1A: "BoxedValueRef",
    }

    @classmethod
    def name(cls, t):
        return cls._NAMES.get(t, f"Unknown(0x{t:02x})")


@dataclass
class EbxFieldDef:
    name: str
    name_hash: int
    type_raw: int
    class_ref: int
    data_offset: int
    second_offset: int

    @property
    def debug_type(self) -> int:
        return (self.type_raw >> 4) & 0x1F


@dataclass
class EbxClassDef:
    name: str
    name_hash: int
    field_index: int
    field_count: int
    alignment: int
    type_raw: int
    size: int
    second_size: int

    @property
    def debug_type(self) -> int:
        return (self.type_raw >> 4) & 0x1F


@dataclass
class EbxInstance:
    class_ref: int
    count: int
    is_exported: bool


@dataclass
class EbxArrayDef:
    offset: int
    count: int
    class_ref: int


@dataclass
class EbxImport:
    file_guid: bytes
    class_guid: bytes


class EbxReader:
    """Schema-less EBX reader that produces generic Python dicts/lists."""

    EBX_VERSION2 = 0x0FB2D1CE
    EBX_VERSION4 = 0x0FB4D1CE

    def __init__(self, data: bytes):
        self.stream = BytesIO(data)
        self.fields: list[EbxFieldDef] = []
        self.classes: list[EbxClassDef] = []
        self.instances: list[EbxInstance] = []
        self.arrays: list[EbxArrayDef] = []
        self.imports: list[EbxImport] = []
        self.objects: list[dict] = []
        self.ref_counts: list[int] = []

        self.strings_offset = 0
        self.strings_len = 0
        self.arrays_offset = 0
        self.data_len = 0
        self.file_guid = b''
        self.magic = 0

        self._parse_header()

    def _read(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.stream.read(size))

    def _read_u8(self): return self._read('<B')[0]
    def _read_u16(self): return self._read('<H')[0]
    def _read_u32(self): return self._read('<I')[0]
    def _read_i32(self): return self._read('<i')[0]
    def _read_u64(self): return self._read('<Q')[0]
    def _read_i64(self): return self._read('<q')[0]
    def _read_f32(self): return self._read('<f')[0]
    def _read_f64(self): return self._read('<d')[0]
    def _read_guid(self): return self.stream.read(16)
    def _read_sha1(self): return self.stream.read(20)

    @property
    def _pos(self):
        return self.stream.tell()

    @_pos.setter
    def _pos(self, v):
        self.stream.seek(v)

    def _pad(self, alignment):
        while self._pos % alignment != 0:
            self._pos += 1

    def _read_null_terminated(self) -> str:
        chars = []
        while True:
            b = self.stream.read(1)
            if b == b'\x00' or len(b) == 0:
                break
            chars.append(b)
        return b''.join(chars).decode('utf-8', errors='replace')

    def _read_sized_string(self, n: int) -> str:
        data = self.stream.read(n)
        # Strip null padding
        idx = data.find(b'\x00')
        if idx >= 0:
            data = data[:idx]
        return data.decode('utf-8', errors='replace')

    def _hash_string(self, s: str) -> int:
        h = 5381
        for c in s:
            h = ((h * 33) ^ ord(c)) & 0xFFFFFFFF
        # Return as signed int32
        if h >= 0x80000000:
            h -= 0x100000000
        return h

    def _parse_header(self):
        self.magic = self._read_u32()
        if self.magic not in (self.EBX_VERSION2, self.EBX_VERSION4):
            raise ValueError(f"Invalid EBX magic: 0x{self.magic:08x}")

        self.strings_offset = self._read_u32()
        strings_and_data_len = self._read_u32()
        guid_count = self._read_u32()
        instance_count = self._read_u16()
        exported_count = self._read_u16()
        unique_class_count = self._read_u16()
        class_type_count = self._read_u16()
        field_type_count = self._read_u16()
        type_names_len = self._read_u16()
        self.strings_len = self._read_u32()
        array_count = self._read_u32()
        self.data_len = self._read_u32()

        self.arrays_offset = self.strings_offset + self.strings_len + self.data_len

        self.file_guid = self._read_guid()

        boxed_values_count = 0
        boxed_values_offset = 0
        if self.magic == self.EBX_VERSION4:
            boxed_values_count = self._read_u32()
            boxed_values_offset = self._read_u32()
            boxed_values_offset += self.strings_offset + self.strings_len
        else:
            self._pad(16)

        # Read imports
        for _ in range(guid_count):
            self.imports.append(EbxImport(
                file_guid=self._read_guid(),
                class_guid=self._read_guid(),
            ))

        # Read type names
        type_names = {}
        type_names_start = self._pos
        while self._pos - type_names_start < type_names_len:
            name = self._read_null_terminated()
            h = self._hash_string(name)
            if h not in type_names:
                type_names[h] = name

        # Read field types
        for _ in range(field_type_count):
            name_hash = self._read_i32()
            type_raw = self._read_u16()
            class_ref = self._read_u16()
            data_offset = self._read_u32()
            second_offset = self._read_u32()
            self.fields.append(EbxFieldDef(
                name=type_names.get(name_hash, f"field_{name_hash:08x}"),
                name_hash=name_hash,
                type_raw=type_raw,
                class_ref=class_ref,
                data_offset=data_offset,
                second_offset=second_offset,
            ))

        # Read class types
        for _ in range(class_type_count):
            name_hash = self._read_i32()
            field_index = self._read_i32()
            field_count = self._read_u8()
            alignment = self._read_u8()
            type_raw = self._read_u16()
            size = self._read_u16()
            second_size = self._read_u16()
            self.classes.append(EbxClassDef(
                name=type_names.get(name_hash, f"class_{name_hash:08x}"),
                name_hash=name_hash,
                field_index=field_index,
                field_count=field_count,
                alignment=alignment,
                type_raw=type_raw,
                size=size,
                second_size=second_size,
            ))

        # Read instances
        tmp_exported = exported_count
        for _ in range(instance_count):
            class_ref = self._read_u16()
            count = self._read_u16()
            is_exported = tmp_exported > 0
            if is_exported:
                tmp_exported -= 1
            self.instances.append(EbxInstance(class_ref, count, is_exported))

        self._pad(16)

        # Read arrays
        for _ in range(array_count):
            offset = self._read_u32()
            count = self._read_u32()
            class_ref = self._read_i32()
            self.arrays.append(EbxArrayDef(offset, count, class_ref))

        self._pad(16)

        # Skip boxed values
        # (we'll handle them if needed)

        # Position to data start
        self._pos = self.strings_offset + self.strings_len

    def read_all_objects(self) -> list[dict]:
        """Read all objects from the EBX data section."""
        self.objects = []
        self.ref_counts = []

        # Create placeholder objects
        for inst in self.instances:
            class_def = self.classes[inst.class_ref]
            for _ in range(inst.count):
                self.objects.append({"_type": class_def.name})
                self.ref_counts.append(0)

        # Read object data
        obj_idx = 0
        for inst in self.instances:
            class_def = self.classes[inst.class_ref]
            for _ in range(inst.count):
                self._pad(class_def.alignment)

                if inst.is_exported:
                    guid = self._read_guid()
                    self.objects[obj_idx]["_guid"] = guid.hex()

                if class_def.alignment != 4:
                    self._pos += 8  # skip 8 bytes (internal offset)

                start_offset = self._pos - 8
                self._read_class(class_def, self.objects[obj_idx], start_offset)
                obj_idx += 1

        return self.objects

    def _read_class(self, class_def: EbxClassDef, obj: dict, start_offset: int):
        """Read a class instance's fields into a dict."""
        for j in range(class_def.field_count):
            field_def = self.fields[class_def.field_index + j]
            field_type = field_def.debug_type

            if field_type == EbxFieldType.INHERITED:
                parent_class = self.classes[field_def.class_ref]
                self._read_class(parent_class, obj, start_offset)
                continue

            # Alignment for certain types
            if field_type in (EbxFieldType.RESOURCE_REF, EbxFieldType.TYPE_REF,
                              EbxFieldType.FILE_REF, EbxFieldType.BOXED_VALUE_REF,
                              EbxFieldType.UINT64, EbxFieldType.INT64, EbxFieldType.FLOAT64):
                self._pad(8)
            elif field_type in (EbxFieldType.ARRAY, EbxFieldType.POINTER):
                self._pad(4)

            if field_type == EbxFieldType.ARRAY:
                array_class = self.classes[field_def.class_ref]
                index = self._read_i32()
                if 0 <= index < len(self.arrays):
                    array_def = self.arrays[index]
                    saved_pos = self._pos
                    self._pos = self.arrays_offset + array_def.offset

                    items = []
                    for _ in range(array_def.count):
                        array_field = self.fields[array_class.field_index]
                        value = self._read_field(array_class, array_field.debug_type, array_field.class_ref)
                        items.append(value)

                    self._pos = saved_pos
                    obj[field_def.name] = items
                else:
                    obj[field_def.name] = []
            else:
                value = self._read_field(class_def, field_type, field_def.class_ref)
                obj[field_def.name] = value

        self._pad(class_def.alignment)

    def _read_field(self, parent_class: Optional[EbxClassDef], field_type: int, class_ref: int) -> Any:
        """Read a single field value."""
        if field_type == EbxFieldType.BOOLEAN:
            return self._read_u8() > 0
        elif field_type == EbxFieldType.INT8:
            return struct.unpack('b', self.stream.read(1))[0]
        elif field_type == EbxFieldType.UINT8:
            return self._read_u8()
        elif field_type == EbxFieldType.INT16:
            return self._read('<h')[0]
        elif field_type == EbxFieldType.UINT16:
            return self._read_u16()
        elif field_type == EbxFieldType.INT32:
            return self._read_i32()
        elif field_type == EbxFieldType.UINT32:
            return self._read_u32()
        elif field_type == EbxFieldType.INT64:
            return self._read_i64()
        elif field_type == EbxFieldType.UINT64:
            return self._read_u64()
        elif field_type == EbxFieldType.FLOAT32:
            return self._read_f32()
        elif field_type == EbxFieldType.FLOAT64:
            return self._read_f64()
        elif field_type == EbxFieldType.GUID:
            return self._read_guid().hex()
        elif field_type == EbxFieldType.SHA1:
            return self._read_sha1().hex()
        elif field_type == EbxFieldType.STRING:
            return self._read_sized_string(32)
        elif field_type == EbxFieldType.CSTRING:
            offset = self._read_u32()
            return self._read_string_at(offset)
        elif field_type == EbxFieldType.RESOURCE_REF:
            return self._read_u64()
        elif field_type == EbxFieldType.FILE_REF:
            str_offset = self._read_u32()
            self._pos += 4  # skip padding
            return self._read_string_at(str_offset)
        elif field_type == EbxFieldType.TYPE_REF:
            str_offset = self._read_u32()
            self._pos += 4
            return self._read_string_at(str_offset)
        elif field_type == EbxFieldType.BOXED_VALUE_REF:
            index = self._read_i32()
            self._pos += 12  # skip rest of boxed value ref
            return {"_boxed_index": index}
        elif field_type == EbxFieldType.STRUCT:
            struct_class = self.classes[class_ref]
            self._pad(struct_class.alignment)
            struct_obj = {"_type": struct_class.name}
            self._read_class(struct_class, struct_obj, self._pos)
            return struct_obj
        elif field_type == EbxFieldType.ENUM:
            return self._read_i32()
        elif field_type == EbxFieldType.POINTER:
            index = self._read_u32()
            if (index >> 31) == 1:
                # External reference
                import_idx = index & 0x7FFFFFFF
                if import_idx < len(self.imports):
                    imp = self.imports[import_idx]
                    return {"_external": imp.file_guid.hex(), "_class": imp.class_guid.hex()}
                return {"_external": f"import_{import_idx}"}
            elif index == 0:
                return None  # null pointer
            else:
                obj_idx = index - 1
                if obj_idx < len(self.objects):
                    self.ref_counts[obj_idx] += 1
                    return {"_ref": int(obj_idx)}
                return {"_ref": int(obj_idx)}
        else:
            return f"<unknown_type_0x{field_type:02x}>"

    def _read_string_at(self, offset: int) -> str:
        """Read a null-terminated string from the strings section."""
        if offset == 0xFFFFFFFF:
            return ""
        saved = self._pos
        self._pos = self.strings_offset + offset
        result = self._read_null_terminated()
        self._pos = saved
        return result

    def get_type_summary(self) -> dict:
        """Return a summary of all types defined in this EBX."""
        result = {}
        for cls in self.classes:
            fields_info = []
            for j in range(cls.field_count):
                f = self.fields[cls.field_index + j]
                fields_info.append({
                    "name": f.name,
                    "type": EbxFieldType.name(f.debug_type),
                    "offset": f.data_offset,
                })
            result[cls.name] = {
                "size": cls.size,
                "alignment": cls.alignment,
                "fields": fields_info,
            }
        return result
