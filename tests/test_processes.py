import time

from reflexmesh.processes import spawn, worker_python


def test_child_tree_cancel_prevents_delayed_descendant_effect(tmp_path):
    child_code = "import pathlib,time; pathlib.Path('ready').write_text('yes'); time.sleep(.5); pathlib.Path('late').write_text('bad')"
    parent_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(10)"
    child = spawn([worker_python(), "-c", parent_code, child_code], cwd=tmp_path)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        child.terminate()
        child.wait(timeout=3)
        time.sleep(0.6)
        assert not (tmp_path / "late").exists()
    finally:
        child.close()


def test_process_output_and_natural_exit(tmp_path):
    with (tmp_path / "stdout").open("w+b") as output:
        child = spawn([worker_python(), "-c", "print(731)"], cwd=tmp_path, stdout=output)
        try:
            assert child.wait(timeout=5) == 0
            output.seek(0)
            assert output.read().strip() == b"731"
        finally:
            child.close()


def test_parallel_launches_preserve_caller_handle_flags(tmp_path):
    import os
    from concurrent.futures import ThreadPoolExecutor

    with (tmp_path / "shared").open("w+b") as output:
        before = os.get_inheritable(output.fileno())

        def launch(value):
            child = spawn([worker_python(), "-c", f"print({value})"], cwd=tmp_path, stdout=output)
            try:
                return child.wait(timeout=10)
            finally:
                child.close()

        with ThreadPoolExecutor(4) as pool:
            assert list(pool.map(launch, range(8))) == [0] * 8
        assert os.get_inheritable(output.fileno()) == before
        output.seek(0)
        assert sorted(map(int, output.read().split())) == list(range(8))
