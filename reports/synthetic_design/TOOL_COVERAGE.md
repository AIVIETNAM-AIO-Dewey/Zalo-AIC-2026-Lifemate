# Tool coverage cho thiết kế synthetic dataset

Chỉ dùng 1236 IDs train cố định. Không dùng nhãn dev hoặc dòng dữ liệu public test.
Public chỉ được đọc catalog. Đây là thống kê cấu trúc, chưa phải kết quả model hay dependency đã kiểm chứng.

Mỗi hàng là một tool. Occurrences đếm số call; samples đếm số hội thoại có call.
`x-ref` là gợi ý nguồn ID; cần đọc runtime và kiểm tra provenance trước khi xây scenario.

| Tool | Kind | Samples / calls | Installed samples | Required | Alternative groups | ID references |
|---|---|---:|---:|---|---|---|
| `alarm.create` | write | 66 / 67 | 396 | time | — | — |
| `alarm.delete` | write | 2 / 2 | 396 | alarm_id | — | alarm_id → alarm.alarms |
| `alarm.list` | read | 7 / 7 | 396 | — | — | — |
| `alarm.update` | write | 6 / 6 | 396 | alarm_id | — | alarm_id → alarm.alarms |
| `appliances.list` | read | 16 / 16 | 88 | — | — | — |
| `appliances.start_cycle` | write | 10 / 11 | 88 | appliance_id, cycle | — | appliance_id → appliances.appliances |
| `appliances.stop` | write | 6 / 6 | 88 | appliance_id | — | appliance_id → appliances.appliances |
| `banking.get_balance` | read | 49 / 49 | 394 | — | — | — |
| `banking.list_beneficiaries` | read | 45 / 45 | 394 | — | — | — |
| `banking.list_transactions` | read | 2 / 2 | 394 | — | — | account_id → banking.accounts |
| `banking.open_savings` | write | 3 / 3 | 394 | from_account_id, amount, term_months | — | from_account_id → banking.accounts |
| `banking.transfer` | write | 47 / 66 | 394 | from_account_id, beneficiary_id, amount | — | from_account_id → banking.accounts; beneficiary_id → banking.beneficiaries |
| `bike_rental.find_bikes` | read | 13 / 13 | 75 | — | — | — |
| `bike_rental.reserve` | write | 22 / 26 | 75 | bike_id | — | bike_id → bike_rental.bikes |
| `bill_payment.list_bills` | read | 89 / 89 | 177 | — | — | — |
| `bill_payment.pay` | write | 62 / 85 | 177 | bill_id, source | — | bill_id → bill_payment.bills |
| `bill_payment.schedule_autopay` | write | 8 / 8 | 177 | type, enabled | — | — |
| `calculator.calculate` | read | 9 / 9 | 97 | expression | — | — |
| `calculator.convert_unit` | read | 10 / 10 | 97 | value, from_unit, to_unit | — | — |
| `calendar.cancel_event` | write | 18 / 18 | 702 | event_id | — | event_id → calendar.events |
| `calendar.create_event` | write | 31 / 32 | 702 | title, start | — | — |
| `calendar.list_events` | read | 81 / 81 | 702 | — | — | — |
| `calendar.update_event` | write | 30 / 30 | 702 | event_id | — | event_id → calendar.events |
| `chat_ott.call` | write | 4 / 4 | 554 | chat_id | — | chat_id → chat_ott.chats |
| `chat_ott.list_chats` | read | 75 / 75 | 554 | — | — | — |
| `chat_ott.read_chat` | read | 9 / 9 | 554 | chat_id | — | chat_id → chat_ott.chats |
| `chat_ott.send_location` | write | 3 / 3 | 554 | chat_id | — | chat_id → chat_ott.chats |
| `chat_ott.send_message` | write | 53 / 56 | 554 | chat_id, content | — | chat_id → chat_ott.chats |
| `contacts.get` | read | 0 / 0 | 696 | contact_id | — | contact_id → contacts.contacts |
| `contacts.search` | read | 176 / 185 | 696 | — | — | — |
| `contacts.update` | write | 9 / 9 | 696 | contact_id | phone / email | contact_id → contacts.contacts |
| `door_lock.grant_guest_code` | write | 7 / 8 | 101 | lock_id, valid_until | — | lock_id → door_lock.locks |
| `door_lock.list_locks` | read | 22 / 22 | 101 | — | — | — |
| `door_lock.set_lock` | write | 21 / 23 | 101 | lock_id, locked | — | lock_id → door_lock.locks |
| `email.archive` | write | 2 / 4 | 294 | email_id, action | — | email_id → email.emails |
| `email.create_draft` | write | 17 / 17 | 294 | to, subject, body | — | — |
| `email.list` | read | 2 / 2 | 294 | — | — | — |
| `email.read` | read | 4 / 4 | 294 | email_id | — | email_id → email.emails |
| `email.reply` | write | 4 / 4 | 294 | email_id, body | — | email_id → email.emails |
| `email.send` | write | 2 / 2 | 294 | to, subject, body | — | — |
| `ewallet.get_balance` | read | 32 / 32 | 430 | — | — | — |
| `ewallet.pay_qr` | write | 4 / 4 | 430 | merchant_id, amount | — | — |
| `ewallet.topup_mobile` | write | 9 / 9 | 430 | phone, amount | — | — |
| `ewallet.transfer` | write | 28 / 29 | 430 | amount | contact_id / phone | contact_id → contacts.contacts |
| `exchange_rate.convert` | read | 8 / 10 | 77 | amount, from_currency, to_currency | — | — |
| `exchange_rate.get_rate` | read | 12 / 12 | 77 | from_currency, to_currency | — | — |
| `fitness.get_activity` | read | 8 / 9 | 86 | — | — | — |
| `fitness.log_workout` | write | 12 / 12 | 86 | type, minutes | — | — |
| `fitness.set_goal` | write | 9 / 9 | 86 | steps | — | — |
| `flights.book` | write | 13 / 13 | 80 | flight_id, passengers | — | flight_id → flights.flights |
| `flights.cancel` | write | 3 / 4 | 80 | booking_id | — | booking_id → flights.bookings |
| `flights.check_in` | write | 7 / 7 | 80 | booking_id | — | booking_id → flights.bookings |
| `flights.my_bookings` | read | 8 / 8 | 80 | — | — | — |
| `flights.search` | read | 5 / 5 | 80 | origin, destination, date | — | — |
| `food_delivery.cancel_order` | write | 5 / 6 | 374 | order_id | — | order_id → food_delivery.orders |
| `food_delivery.get_menu` | read | 20 / 20 | 374 | restaurant_id | — | restaurant_id → food_delivery.restaurants |
| `food_delivery.list_restaurants` | read | 23 / 23 | 374 | — | — | — |
| `food_delivery.order_history` | read | 13 / 13 | 374 | — | — | — |
| `food_delivery.place_order` | write | 24 / 25 | 374 | restaurant_id, items | — | restaurant_id → food_delivery.restaurants |
| `fuel_price.get_price` | read | 9 / 9 | 241 | — | — | — |
| `gold_price.get_price` | read | 27 / 27 | 279 | — | — | — |
| `home_automation.create_rule` | write | 17 / 17 | 87 | device_id, action, time | — | device_id → smart_home.devices |
| `home_automation.delete_rule` | write | 13 / 16 | 87 | rule_id | — | rule_id → home_automation.rules |
| `home_automation.list_rules` | read | 10 / 10 | 87 | — | — | — |
| `home_energy.get_usage` | read | 16 / 16 | 91 | period | — | — |
| `home_energy.set_alert` | write | 10 / 10 | 91 | threshold_kwh | — | — |
| `home_security.acknowledge_alert` | write | 7 / 10 | 111 | alert_id | — | alert_id → home_security.alerts |
| `home_security.arm` | write | 2 / 2 | 111 | mode | — | — |
| `home_security.get_camera` | read | 6 / 6 | 111 | — | — | — |
| `home_security.list_alerts` | read | 21 / 21 | 111 | — | — | — |
| `household_chores.assign` | write | 18 / 22 | 96 | assignee, chore_id | — | chore_id → household_chores.chores |
| `household_chores.list_chores` | read | 16 / 16 | 96 | — | — | — |
| `learning.enroll` | write | 9 / 9 | 88 | course_id | — | course_id → learning.courses |
| `learning.list_courses` | read | 17 / 17 | 88 | — | — | — |
| `learning.log_progress` | write | 8 / 8 | 88 | course_id, minutes | — | course_id → learning.courses |
| `learning.schedule_lesson` | write | 8 / 8 | 88 | course_id, time | — | course_id → learning.courses |
| `library.reserve` | write | 11 / 11 | 84 | book_id | — | book_id → library.books |
| `library.search` | read | 10 / 10 | 84 | — | — | — |
| `lottery.check_ticket` | read | 8 / 11 | 58 | province, date, number | — | — |
| `lottery.get_results` | read | 9 / 11 | 58 | province | — | — |
| `loyalty.get_points` | read | 15 / 15 | 79 | — | — | — |
| `loyalty.list_vouchers` | read | 5 / 5 | 79 | — | — | — |
| `loyalty.redeem` | write | 13 / 17 | 79 | program, reward_id | — | — |
| `maps.get_route` | read | 15 / 16 | 237 | origin, destination | — | — |
| `maps.save_place` | write | 9 / 10 | 237 | place_id, label | — | place_id → maps.places |
| `maps.search_place` | read | 9 / 10 | 237 | query | — | — |
| `medication.add` | write | 5 / 5 | 272 | name, times | — | — |
| `medication.list` | read | 14 / 14 | 272 | — | — | — |
| `medication.log_taken` | write | 5 / 7 | 272 | schedule_id | — | schedule_id → medication.schedules |
| `medication.remove` | write | 5 / 6 | 272 | schedule_id | — | schedule_id → medication.schedules |
| `music.add_to_playlist` | write | 4 / 5 | 274 | track_id, playlist_id | — | track_id → music.tracks; playlist_id → music.playlists |
| `music.control` | write | 3 / 3 | 274 | action | — | — |
| `music.list_tracks` | read | 11 / 11 | 274 | — | — | — |
| `music.play` | write | 14 / 14 | 274 | — | track_id / playlist_id | track_id → music.tracks; playlist_id → music.playlists |
| `news.headlines` | read | 3 / 3 | 116 | — | — | — |
| `news.save` | write | 11 / 15 | 116 | article_id | — | article_id → news.articles |
| `news.search` | read | 20 / 21 | 116 | query | — | — |
| `notes.append` | write | 4 / 4 | 454 | note_id, content | — | note_id → notes.notes |
| `notes.create` | write | 23 / 23 | 454 | content | — | — |
| `notes.list` | read | 8 / 8 | 454 | — | — | — |
| `package_tracking.reschedule_delivery` | write | 5 / 5 | 81 | tracking_id, time | — | — |
| `package_tracking.schedule_pickup` | write | 13 / 13 | 81 | address, date | — | — |
| `package_tracking.track` | read | 8 / 8 | 81 | — | — | — |
| `phone.call` | write | 26 / 26 | 405 | — | contact_id / phone_number | contact_id → contacts.contacts |
| `phone.recent_calls` | read | 3 / 3 | 405 | — | — | — |
| `podcast_radio.play` | write | 4 / 4 | 107 | — | episode_id / station | episode_id → podcast_radio.episodes |
| `podcast_radio.queue` | write | 11 / 12 | 107 | episode_id | — | episode_id → podcast_radio.episodes |
| `podcast_radio.search` | read | 17 / 17 | 107 | query | — | — |
| `podcast_radio.subscribe` | write | 5 / 5 | 107 | show_id | — | show_id → podcast_radio.shows |
| `reminders.complete` | write | 3 / 4 | 705 | reminder_id | — | reminder_id → reminders.reminders |
| `reminders.create` | write | 78 / 79 | 705 | content, time | — | — |
| `reminders.delete` | write | 2 / 2 | 705 | reminder_id | — | reminder_id → reminders.reminders |
| `reminders.list` | read | 9 / 9 | 705 | — | — | — |
| `ride_hailing.book` | write | 32 / 32 | 395 | pickup, dropoff, vehicle | — | — |
| `ride_hailing.cancel` | write | 6 / 6 | 395 | ride_id | — | ride_id → ride_hailing.rides |
| `ride_hailing.get_quotes` | read | 1 / 1 | 395 | pickup, dropoff | — | — |
| `ride_hailing.get_ride` | read | 3 / 3 | 395 | — | — | ride_id → ride_hailing.rides |
| `robot_vacuum.get_status` | read | 13 / 13 | 92 | — | — | — |
| `robot_vacuum.start` | write | 15 / 15 | 92 | — | — | — |
| `robot_vacuum.stop` | write | 13 / 13 | 92 | — | — | — |
| `shopping_list.add` | write | 13 / 13 | 82 | name | — | — |
| `shopping_list.check` | write | 4 / 7 | 82 | item_id | — | item_id → shopping_list.items |
| `shopping_list.get` | read | 7 / 7 | 82 | — | — | — |
| `shopping_list.remove` | write | 4 / 4 | 82 | item_id | — | item_id → shopping_list.items |
| `smart_home.climate_set` | write | 34 / 34 | 522 | device_id, mode | — | device_id → smart_home.devices |
| `smart_home.fan_set` | write | 6 / 7 | 522 | device_id | — | device_id → smart_home.devices |
| `smart_home.get_status` | read | 91 / 91 | 522 | — | — | — |
| `smart_home.light_set` | write | 79 / 107 | 522 | device_id | — | device_id → smart_home.devices |
| `smart_home.scene_run` | write | 1 / 1 | 522 | scene_id | — | scene_id → smart_home.scenes |
| `smart_home.switch_set` | write | 3 / 3 | 522 | device_id, power | — | device_id → smart_home.devices |
| `sms.read_recent` | read | 6 / 6 | 329 | — | — | contact_id → contacts.contacts |
| `sms.send` | write | 59 / 60 | 329 | content | contact_id / phone_number | contact_id → contacts.contacts |
| `subscriptions.cancel` | write | 22 / 23 | 100 | subscription_id | — | subscription_id → subscriptions.subscriptions |
| `subscriptions.list` | read | 11 / 11 | 100 | — | — | — |
| `sys.ask_missing_info` | meta | 179 / 179 | 1236 | tool_name, params | — | — |
| `tasks.add` | write | 32 / 33 | 443 | title | — | — |
| `tasks.complete` | write | 10 / 11 | 443 | task_id | — | task_id → tasks.tasks |
| `tasks.list` | read | 12 / 12 | 443 | — | — | — |
| `tv.control` | write | 5 / 5 | 273 | action | — | — |
| `tv.guide` | read | 3 / 3 | 273 | — | — | — |
| `tv.record` | write | 1 / 1 | 273 | program_id | — | program_id → tv.programs |
| `tv.tune` | write | 2 / 2 | 273 | channel | — | — |
| `video_meeting.create` | write | 10 / 10 | 99 | title, start | — | — |
| `video_meeting.join` | write | 7 / 7 | 99 | meeting_id | — | meeting_id → video_meeting.meetings |
| `video_meeting.list_meetings` | read | 13 / 13 | 99 | — | — | — |
| `weather.get_current` | read | 14 / 14 | 514 | — | — | — |
| `weather.get_forecast` | read | 48 / 48 | 514 | date | — | — |

## Catalog public có thêm

Chỉ là danh sách capability công khai để lập backlog. Pilot chưa sinh dữ liệu cho các tool này.

`ecommerce.add_to_cart`, `ecommerce.cancel_order`, `ecommerce.checkout`, `ecommerce.order_status`, `ecommerce.request_return`, `ecommerce.search`, `event_tickets.book`, `event_tickets.search`, `grocery.add_to_cart`, `grocery.checkout`, `grocery.search`, `habit_tracker.list_habits`, `habit_tracker.log`, `pantry.consume`, `pantry.list_items`, `parking.end_session`, `parking.find`, `parking.list_sessions`, `parking.reserve`, `public_transit.find_routes`, `public_transit.next_departures`, `public_transit.topup_card`, `recipes.get`, `recipes.save`, `recipes.search`, `stocks.get_portfolio`, `stocks.get_quote`, `stocks.place_order`, `stocks.set_price_alert`, `train_bus_tickets.book`, `train_bus_tickets.cancel`, `train_bus_tickets.my_bookings`, `train_bus_tickets.search`, `translator.translate`

## Schema tham số thay đổi giữa hai catalog

Không có.

Chi tiết counts trước/sát write và checksum nguồn nằm trong `tool_inventory.json` cùng thư mục.
