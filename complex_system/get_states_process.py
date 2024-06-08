import traceback

from global_env import ROBOTS
import time
import requests


def get_statuses(status_dict):
    robots = [robot['url'] for robot in ROBOTS]
    api_endpoint = 'api/v2/robot/state'

    count = 0
    while True:
        time.sleep(5)

        count += 1

        flag = False
        for i, robot_url in enumerate(robots):
            final_result = {
                'battery': -1,
                'flag': -1
            }
            try:
                try:
                    if count % 10000 == 0:
                        response = requests.put(f'{robot_url}/api/v2/robot/capabilities/MapResetCapability', data={"action": "reset"})
                        flag = True
                except:
                    pass

                for _ in range(10):
                    try:
                        response = requests.get(f'{robot_url}{api_endpoint}')
                        data = response.json()
                        break
                    except:
                        print("Ошибка в статусе", ROBOTS[i]['Name'], response)
                    time.sleep(1)

                # Переменные для промежуточного словаря
                intermediate_result = {}

                for attribute in data['attributes']:
                    if attribute['__class'] == 'StatusStateAttribute':
                        intermediate_result['value'] = attribute['value']
                    if attribute['__class'] == 'BatteryStateAttribute':
                        intermediate_result['battery'] = attribute['level']
                        intermediate_result['flag'] = attribute['flag']

                # Преобразование словаря по условиям
                final_result = {
                    'battery': intermediate_result.get('battery'),
                    'flag': intermediate_result.get('flag')
                }

                value = intermediate_result.get('value')
                if value == 'docked' and final_result['flag'] == 'charging':
                    final_result['flag'] = 'charging'
                elif value == 'returning' and final_result['flag'] == 'discharging':
                    final_result['flag'] = 'go_home'
                else:
                    final_result['flag'] = 'discharging'

                status_dict[i] = final_result
            except Exception as e:
                print(f"Except 4: {traceback.format_exc()}")
                status_dict[i] = final_result

            if flag:
                count = 0
                flag = False


if __name__ == '__main__':
    main = {}
    get_statuses(main)
