import json
import json
from typing import Generator, List, Tuple

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.data_manager import data_manager

client = TestClient(app)


def _iter_sse(response) -> Generator[Tuple[str, str], None, None]:
  buffer = ''
  for chunk in response.iter_text():
    buffer += chunk
    while '\n\n' in buffer:
      block, buffer = buffer.split('\n\n', 1)
      event_type = 'message'
      data_lines: List[str] = []
      for line in block.splitlines():
        if line.startswith(':'):
          continue
        if line.startswith('event:'):
          event_type = line[len('event:') :].strip()
        elif line.startswith('data:'):
          data_lines.append(line[len('data:') :].strip())
      yield event_type, '\n'.join(data_lines)


def test_train_stream_emits_history_and_result():
  metadata = data_manager.load_sample_dataset('titanic')
  payload = {
    'dataset_id': metadata.dataset_id,
    'target_column': 'Survived',
    'task_type': 'classification',
    'algorithm': 'logistic_regression',
    'hyperparameters': {'max_iter': 50},
  }

  params = {'payload': json.dumps(payload)}

  with client.stream('GET', '/model/train/stream', params=params) as response:
    assert response.status_code == 200
    events = []
    for event_type, data in _iter_sse(response):
      events.append((event_type, data))
      if event_type == 'result':
        break

  event_types = [event for event, _ in events]
  assert 'history' in event_types
  assert 'result' in event_types

  result_data = next(data for event, data in events if event == 'result')
  result_payload = json.loads(result_data).get('payload', {})

  assert result_payload.get('model_id')
  assert result_payload.get('split_id')
  assert result_payload.get('metrics', {}).get('validation')
  assert isinstance(result_payload.get('history'), list)
