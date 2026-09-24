"""Owned process trees: suspended Windows Job admission and POSIX process groups.

This is lifecycle containment for cooperating executables, not a privilege sandbox.
Windows Job semantics: https://learn.microsoft.com/windows/win32/procthread/job-objects
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def spawn(argv, *, cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=None):
    """Start immutable argv in an owned tree, before any target code can execute."""
    if not argv or not Path(argv[0]).is_absolute():
        raise ValueError("process executable must be absolute")
    validate_executable(argv[0])
    if os.name == "nt":
        return _WindowsProcess(argv, cwd, stdout, stderr, env)
    return _PosixProcess(argv, cwd, stdout, stderr, env)


def worker_python() -> str:
    """Direct current base interpreter for stdlib-only owned workers.

    Store-backed Windows venv redirectors use an activation service that can leave
    Jobs. The direct base image retains Job membership. This explicitly selects
    the base environment, so callers must not expect virtualenv-only packages.
    """
    if os.name == "nt":
        direct = Path(sys.base_prefix) / "python.exe"
        if direct.is_file():
            return str(direct)
    return sys.executable


def validate_executable(executable) -> None:
    path = Path(executable)
    if not path.is_file():
        raise ValueError("process executable must be a regular existing image")
    if os.name == "nt":
        if getattr(path.lstat(), "st_reparse_tag", 0) == 0x8000001B:
            raise ValueError("App Execution aliases cannot be supervised as a Job tree")
        config = path.parent.parent / "pyvenv.cfg"
        if (
            path.parent.name.lower() == "scripts"
            and config.is_file()
            and "windowsapps" in config.read_text().lower()
        ):
            raise ValueError(
                "Store-backed virtualenv redirector escapes Job admission; use worker_python() for stdlib workers or a regular Python installation"
            )


class _PosixProcess:
    containment = "posix-process-group"

    def __init__(self, argv, cwd, stdout, stderr, env):
        self._process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            shell=False,
            start_new_session=True,
        )
        self.pid = self._process.pid
        self.args = argv

    @property
    def returncode(self):
        return self._process.returncode

    def poll(self):
        return self._process.poll()

    def wait(self, timeout=None):
        return self._process.wait(timeout=timeout)

    def terminate(self):
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self):
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def close(self):
        self.kill()
        self._process.wait(timeout=5)


if os.name == "nt":
    import ctypes as ct
    import msvcrt
    from ctypes import wintypes as wt

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    HANDLE, DWORD, SIZE_T = wt.HANDLE, wt.DWORD, ct.c_size_t

    class Startup(ct.Structure):
        _fields_ = [
            ("cb", DWORD),
            ("lpReserved", wt.LPWSTR),
            ("lpDesktop", wt.LPWSTR),
            ("lpTitle", wt.LPWSTR),
            ("dwX", DWORD),
            ("dwY", DWORD),
            ("dwXSize", DWORD),
            ("dwYSize", DWORD),
            ("dwXCountChars", DWORD),
            ("dwYCountChars", DWORD),
            ("dwFillAttribute", DWORD),
            ("dwFlags", DWORD),
            ("wShowWindow", wt.WORD),
            ("cbReserved2", wt.WORD),
            ("lpReserved2", ct.POINTER(ct.c_byte)),
            ("hStdInput", HANDLE),
            ("hStdOutput", HANDLE),
            ("hStdError", HANDLE),
        ]

    class StartupEx(ct.Structure):
        _fields_ = [("StartupInfo", Startup), ("lpAttributeList", ct.c_void_p)]

    class ProcessInfo(ct.Structure):
        _fields_ = [
            ("hProcess", HANDLE),
            ("hThread", HANDLE),
            ("dwProcessId", DWORD),
            ("dwThreadId", DWORD),
        ]

    class BasicLimits(ct.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ct.c_int64),
            ("PerJobUserTimeLimit", ct.c_int64),
            ("LimitFlags", DWORD),
            ("MinimumWorkingSetSize", SIZE_T),
            ("MaximumWorkingSetSize", SIZE_T),
            ("ActiveProcessLimit", DWORD),
            ("Affinity", SIZE_T),
            ("PriorityClass", DWORD),
            ("SchedulingClass", DWORD),
        ]

    class IoCounters(ct.Structure):
        _fields_ = [
            (name, ct.c_uint64)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class ExtendedLimits(ct.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", SIZE_T),
            ("JobMemoryLimit", SIZE_T),
            ("PeakProcessMemoryUsed", SIZE_T),
            ("PeakJobMemoryUsed", SIZE_T),
        ]

    class Accounting(ct.Structure):
        _fields_ = [
            ("TotalUserTime", ct.c_int64),
            ("TotalKernelTime", ct.c_int64),
            ("ThisPeriodTotalUserTime", ct.c_int64),
            ("ThisPeriodTotalKernelTime", ct.c_int64),
            ("TotalPageFaultCount", DWORD),
            ("TotalProcesses", DWORD),
            ("ActiveProcesses", DWORD),
            ("TotalTerminatedProcesses", DWORD),
        ]

    def _signature(name, args, result=wt.BOOL):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
        return function

    CreateJob = _signature("CreateJobObjectW", [ct.c_void_p, wt.LPCWSTR], HANDLE)
    SetJob = _signature("SetInformationJobObject", [HANDLE, ct.c_int, ct.c_void_p, DWORD])
    QueryJob = _signature(
        "QueryInformationJobObject", [HANDLE, ct.c_int, ct.c_void_p, DWORD, ct.c_void_p]
    )
    AssignJob = _signature("AssignProcessToJobObject", [HANDLE, HANDLE])
    TerminateJob = _signature("TerminateJobObject", [HANDLE, wt.UINT])
    Close = _signature("CloseHandle", [HANDLE])
    CurrentProcess = _signature("GetCurrentProcess", [], HANDLE)
    Duplicate = _signature(
        "DuplicateHandle", [HANDLE, HANDLE, HANDLE, ct.POINTER(HANDLE), DWORD, wt.BOOL, DWORD]
    )
    Resume = _signature("ResumeThread", [HANDLE], DWORD)
    ExitCode = _signature("GetExitCodeProcess", [HANDLE, ct.POINTER(DWORD)])
    Wait = _signature("WaitForSingleObject", [HANDLE, DWORD], DWORD)
    Terminate = _signature("TerminateProcess", [HANDLE, wt.UINT])
    Create = _signature(
        "CreateProcessW",
        [
            wt.LPCWSTR,
            wt.LPWSTR,
            ct.c_void_p,
            ct.c_void_p,
            wt.BOOL,
            DWORD,
            ct.c_void_p,
            wt.LPCWSTR,
            ct.c_void_p,
            ct.POINTER(ProcessInfo),
        ],
    )
    Initialize = _signature(
        "InitializeProcThreadAttributeList", [ct.c_void_p, DWORD, DWORD, ct.POINTER(SIZE_T)]
    )
    Update = _signature(
        "UpdateProcThreadAttribute",
        [ct.c_void_p, DWORD, SIZE_T, ct.c_void_p, SIZE_T, ct.c_void_p, ct.c_void_p],
    )
    Delete = _signature("DeleteProcThreadAttributeList", [ct.c_void_p], None)

    def checked(result):
        if not result:
            raise ct.WinError(ct.get_last_error())
        return result

    class _WindowsProcess:
        containment = "windows-job"

        def __init__(self, argv, cwd, stdout, stderr, env):
            self.args, self.returncode = list(argv), None
            self._job, self._process = None, None
            self._closed = False
            null = open(os.devnull, "r+b", buffering=0)  # noqa: SIM115 - closed in the constructor's unconditional finally
            outputs = [
                null,
                null if stdout == subprocess.DEVNULL else stdout,
                null if stderr == subprocess.DEVNULL else stderr,
            ]
            original_handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in outputs]
            duplicates = {}
            process = ProcessInfo()
            attribute_buffer = None
            initialized = False
            try:
                current = CurrentProcess()
                for handle in set(original_handles):
                    duplicate = HANDLE()
                    checked(Duplicate(current, handle, current, ct.byref(duplicate), 0, True, 2))
                    duplicates[handle] = duplicate.value
                handles = [duplicates[handle] for handle in original_handles]
                self._job = checked(CreateJob(None, None))
                limits = ExtendedLimits()
                limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway
                checked(SetJob(self._job, 9, ct.byref(limits), ct.sizeof(limits)))
                startup = StartupEx()
                startup.StartupInfo.cb = ct.sizeof(startup)
                startup.StartupInfo.dwFlags = 0x100 | 0x1  # explicit std handles + hidden window
                (
                    startup.StartupInfo.hStdInput,
                    startup.StartupInfo.hStdOutput,
                    startup.StartupInfo.hStdError,
                ) = handles
                size = SIZE_T()
                Initialize(None, 1, 0, ct.byref(size))
                attribute_buffer = ct.create_string_buffer(size.value)
                startup.lpAttributeList = ct.cast(attribute_buffer, ct.c_void_p)
                checked(Initialize(startup.lpAttributeList, 1, 0, ct.byref(size)))
                initialized = True
                inherit = (HANDLE * len(set(handles)))(*sorted(set(handles)))
                checked(
                    Update(
                        startup.lpAttributeList, 0, 0x20002, inherit, ct.sizeof(inherit), None, None
                    )
                )
                environment = (
                    None
                    if env is None
                    else ct.create_unicode_buffer(
                        "\0".join(
                            f"{k}={v}"
                            for k, v in sorted(env.items(), key=lambda item: item[0].upper())
                        )
                        + "\0\0"
                    )
                )
                command = ct.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in argv]))
                flags = (
                    0x4 | 0x08000000 | 0x80000 | 0x400
                )  # suspended, hidden, extended startup, Unicode env
                checked(
                    Create(
                        str(argv[0]),
                        command,
                        None,
                        None,
                        True,
                        flags,
                        environment,
                        str(cwd),
                        ct.byref(startup),
                        ct.byref(process),
                    )
                )
                self._process, self.pid = process.hProcess, process.dwProcessId
                checked(AssignJob(self._job, process.hProcess))
                if Resume(process.hThread) == 0xFFFFFFFF:
                    raise ct.WinError(ct.get_last_error())
            except BaseException:
                if process.hProcess:
                    Terminate(process.hProcess, 1)
                if self._job:
                    Close(self._job)
                if self._process:
                    Close(self._process)
                self._closed = True
                raise
            finally:
                for duplicate in duplicates.values():
                    Close(duplicate)
                if process.hThread:
                    Close(process.hThread)
                if initialized:
                    Delete(ct.cast(attribute_buffer, ct.c_void_p))
                null.close()

        def poll(self):
            if self._closed:
                return self.returncode
            accounting = Accounting()
            checked(QueryJob(self._job, 1, ct.byref(accounting), ct.sizeof(accounting), None))
            if accounting.ActiveProcesses:
                return None
            if Wait(self._process, 0) != 0:
                return None
            value = DWORD()
            checked(ExitCode(self._process, ct.byref(value)))
            self.returncode = value.value
            return self.returncode

        def wait(self, timeout=None):
            started = time.monotonic()
            while self.poll() is None:
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise subprocess.TimeoutExpired(self.args, timeout)
                time.sleep(0.002)
            return self.returncode

        def terminate(self):
            if not self._closed:
                checked(TerminateJob(self._job, 1))

        kill = terminate

        def close(self):
            if not self._closed:
                try:
                    self.terminate()
                    self.wait(timeout=5)
                finally:
                    Close(self._process)
                    Close(self._job)
                    self._closed = True

        def __del__(self):
            if not getattr(self, "_closed", True):
                # Last-handle close terminates the entire admitted tree.
                Close(self._job)
                Close(self._process)
